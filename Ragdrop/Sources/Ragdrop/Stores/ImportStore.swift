import Foundation
import Observation

@MainActor
@Observable
final class ImportStore {
    var jobs: [ImportJob] = [] {
        didSet { persistJobs() }
    }
    let isIsolated: Bool
    var showingFileImporter = false
    var isRunning = false
    var message = "Drop PDFs to begin."
    @ObservationIgnored private var activeTask: Task<Void, Never>?
    @ObservationIgnored private let queueStoreURL: URL

    init(queueStoreURL: URL = ImportStore.defaultQueueStoreURL, isIsolated: Bool = false) {
        self.isIsolated = isIsolated
        self.queueStoreURL = queueStoreURL
        jobs = Self.loadJobs(from: queueStoreURL).map(Self.recoverInterruptedJob)
        if !jobs.isEmpty {
            message = "Previous batch restored. You can resume processing."
        }
    }

    func addFiles(_ urls: [URL]) {
        let existing = Set(jobs.map { $0.fileURL.standardizedFileURL })
        let additions = urls
            .map(\.standardizedFileURL)
            .filter { $0.pathExtension.lowercased() == "pdf" && !existing.contains($0) }
            .map { ImportJob(fileURL: $0) }
        jobs.append(contentsOf: additions)
        if !additions.isEmpty {
            message = additions.count == 1
                ? "One PDF is ready."
                : "\(additions.count) PDFs are ready."
        }
    }

    func addZoteroDocuments(_ documents: [ZoteroPDF]) {
        var urls = Set(jobs.map { $0.fileURL.standardizedFileURL })
        var keys = Set(jobs.compactMap { $0.metadata?.zoteroAttachmentKey })
        var hashes = Set(jobs.compactMap(\.fingerprint))
        var additions: [ImportJob] = []
        for document in documents {
            guard !document.isIndexed, !urls.contains(document.fileURL.standardizedFileURL),
                  !keys.contains(document.attachmentKey),
                  !(document.fingerprint.map { hashes.contains($0) } ?? false) else { continue }
            var job = ImportJob(fileURL: document.fileURL, metadata: document.importMetadata)
            job.fingerprint = document.fingerprint
            additions.append(job)
            urls.insert(job.fileURL)
            keys.insert(document.attachmentKey)
            if let hash = document.fingerprint { hashes.insert(hash) }
        }
        jobs.append(contentsOf: additions)
        if !additions.isEmpty {
            message = additions.count == 1
                ? "One Zotero PDF is ready."
                : "\(additions.count) Zotero PDFs are ready."
        }
    }

    func removeJobs(at offsets: IndexSet) {
        guard !isRunning else { return }
        for index in offsets.sorted(by: >) where jobs.indices.contains(index) {
            removeJob(at: index)
        }
        updateMessageAfterRemoval()
    }

    func remove(_ id: UUID) {
        guard !isRunning, let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        removeJob(at: index)
        updateMessageAfterRemoval()
    }

    func clearCompleted() {
        guard !isRunning else { return }
        jobs.removeAll { [.completed, .duplicate, .rejected].contains($0.stage) }
    }

    func start() {
        guard !isIsolated, !isRunning else { return }
        isRunning = true
        if jobs.contains(where: { $0.stage == .readyForIndexing }) {
            activeTask = Task { await importApproved() }
        } else if jobs.contains(where: { $0.stage == .queued || $0.stage == .failed }) {
            activeTask = Task { await prepareQueue() }
        } else {
            isRunning = false
        }
    }

    func retry(_ id: UUID) {
        guard !isIsolated, !isRunning, let index = jobs.firstIndex(where: { $0.id == id }),
              jobs[index].stage == .failed else { return }
        jobs[index].errorDetails = nil
        update(index, stage: .queued, detail: "Ready to retry")
        isRunning = true
        activeTask = Task { await prepareQueue(only: Set([id])) }
    }

    func cancelConversion() {
        guard isRunning, jobs.contains(where: {
            $0.stage == .checkingDuplicate || $0.stage == .converting
        }) else { return }
        message = "Canceling conversion…"
        activeTask?.cancel()
    }

    var canCancelConversion: Bool {
        isRunning && jobs.contains(where: {
            $0.stage == .checkingDuplicate || $0.stage == .converting
        })
    }

    func approve(_ id: UUID) {
        guard !isRunning, let index = jobs.firstIndex(where: { $0.id == id }),
              jobs[index].stage == .awaitingReview else { return }
        update(index, stage: .readyForIndexing, detail: "Ready to send to the server")
        message = "Markdown approved. You can add it to Ragdoc."
    }

    func reject(_ id: UUID) {
        guard !isRunning, let index = jobs.firstIndex(where: { $0.id == id }),
              jobs[index].stage == .awaitingReview else { return }
        removeTemporaryFiles(for: jobs[index])
        jobs[index].artifactURL = nil
        jobs[index].metadataURL = nil
        jobs[index].visualArtifactBundleURL = nil
        update(index, stage: .rejected, detail: "Markdown rejected")
        message = "The document was rejected."
    }

    func approveAll() {
        guard !isRunning else { return }
        for index in jobs.indices where jobs[index].stage == .awaitingReview {
            update(index, stage: .readyForIndexing, detail: "Ready to send to the server")
        }
        message = "Markdown files approved. You can start adding them."
    }

    private func prepareQueue(only selectedIDs: Set<UUID>? = nil) async {
        let configuration = PipelineConfiguration.current()
        let pipeline = ImportPipeline(configuration: configuration)
        let candidates = jobs.indices.filter {
            (jobs[$0].stage == .queued || jobs[$0].stage == .failed)
                && (selectedIDs == nil || selectedIDs?.contains(jobs[$0].id) == true)
        }
        let candidateSet = Set(candidates)
        var fingerprintsSeen = Set(
            jobs.indices
                .filter { !candidateSet.contains($0) && jobs[$0].stage.ownsFingerprint }
                .compactMap { jobs[$0].fingerprint }
        )

        for index in candidates {
            if Task.isCancelled { break }
            do {
                jobs[index].errorDetails = nil
                update(index, stage: .checkingDuplicate, detail: "Comparing the PDF fingerprint")
                message = "Checking for duplicates of \(jobs[index].displayName)…"
                let duplicate = try await pipeline.duplicateFilename(for: jobs[index].fileURL)
                jobs[index].fingerprint = duplicate.fingerprint
                if fingerprintsSeen.contains(duplicate.fingerprint) {
                    update(index, stage: .duplicate, detail: "The same PDF is already in this batch")
                    continue
                }
                if let filename = duplicate.filename {
                    update(index, stage: .duplicate, detail: "Already present: \(filename)")
                    continue
                }

                jobs[index].converterName = configuration.converterKind.commandLabel
                update(index, stage: .converting, detail: "PDF analysis · \(configuration.converterKind.commandLabel)")
                message = "Converting \(jobs[index].displayName)…"
                let artifact = try await pipeline.convert(
                    jobs[index].fileURL,
                    fingerprint: duplicate.fingerprint,
                    metadata: jobs[index].metadata
                )
                jobs[index].artifactURL = artifact.markdownURL
                jobs[index].metadataURL = artifact.metadataURL
                jobs[index].visualArtifactBundleURL = artifact.artifactBundleURL
                jobs[index].visualArtifactCount = artifact.artifactCount
                fingerprintsSeen.insert(duplicate.fingerprint)
                let visuals = artifact.artifactCount == 1
                    ? "1 table or figure detected"
                    : "\(artifact.artifactCount) tables and figures detected"
                update(index, stage: .awaitingReview, detail: "Markdown ready · \(visuals)")
            } catch is CancellationError {
                jobs[index].errorDetails = "Conversion was canceled by the user."
                update(index, stage: .failed, detail: PipelineError.cancelled.localizedDescription)
                break
            } catch {
                jobs[index].errorDetails = (error as? PipelineError)?.diagnosticDetails
                    ?? error.localizedDescription
                update(index, stage: .failed, detail: error.localizedDescription)
            }
        }

        isRunning = false
        activeTask = nil
        if Task.isCancelled {
            message = "Conversion canceled. You can retry this PDF."
            return
        }
        let ready = jobs.filter { $0.stage == .awaitingReview }.count
        message = ready == 1
            ? "Markdown is ready. Review it before adding."
            : "\(ready) Markdown files are ready to review."
    }

    private func importApproved() async {
        defer {
            isRunning = false
            activeTask = nil
        }
        let pipeline = ImportPipeline(configuration: .current())
        let approved = jobs.indices.filter { jobs[$0].stage == .readyForIndexing }
        var transferred: [Int] = []

        for index in approved {
            jobs[index].errorDetails = nil
            guard let markdownURL = jobs[index].artifactURL else {
                jobs[index].errorDetails = "The temporary Markdown file expected by Ragdrop no longer exists."
                update(index, stage: .failed, detail: "The temporary Markdown file is missing.")
                continue
            }
            let artifact = ConversionArtifact(
                sourceURL: jobs[index].fileURL,
                markdownURL: markdownURL,
                remoteFilename: markdownURL.lastPathComponent,
                metadataURL: jobs[index].metadataURL,
                artifactBundleURL: jobs[index].visualArtifactBundleURL,
                artifactCount: jobs[index].visualArtifactCount
            )
            update(index, stage: .transferring, detail: "Atomically copying Markdown")
            do {
                try await pipeline.transfer(artifact)
                transferred.append(index)
                update(index, stage: .readyForIndexing, detail: "Transferred, waiting for batch indexing")
            } catch {
                jobs[index].errorDetails = (error as? PipelineError)?.diagnosticDetails
                    ?? error.localizedDescription
                update(index, stage: .readyForIndexing, detail: "Transfer needs retrying: \(error.localizedDescription)")
            }
        }

        guard !transferred.isEmpty else {
            isRunning = false
            message = "No Markdown files could be transferred."
            return
        }

        for index in transferred {
            update(index, stage: .indexing, detail: "Creating embeddings")
        }
        message = "Indexing the batch in Ragdoc…"

        do {
            let sources = transferred.compactMap { jobs[$0].artifactURL?.lastPathComponent }
            try await pipeline.indexAll(sources: sources)
        } catch {
            let diagnostic = (error as? PipelineError)?.diagnosticDetails
                ?? error.localizedDescription
            for index in transferred {
                jobs[index].errorDetails = diagnostic
                update(index, stage: .readyForIndexing, detail: "Indexing needs retrying: \(error.localizedDescription)")
            }
            isRunning = false
            message = "Indexing failed."
            return
        }

        for index in transferred {
            guard let markdownURL = jobs[index].artifactURL else { continue }
            let artifact = ConversionArtifact(
                sourceURL: jobs[index].fileURL,
                markdownURL: markdownURL,
                remoteFilename: markdownURL.lastPathComponent,
                metadataURL: jobs[index].metadataURL,
                artifactBundleURL: jobs[index].visualArtifactBundleURL,
                artifactCount: jobs[index].visualArtifactCount
            )
            update(index, stage: .verifying, detail: "Counting passages in the canonical database")
            do {
                let chunks = try await pipeline.verify(artifact)
                let visuals = artifact.artifactCount > 0 ? " · \(artifact.artifactCount) visual \(artifact.artifactCount == 1 ? "item" : "items")" : ""
                update(index, stage: .completed, detail: "\(chunks) indexed \(chunks == 1 ? "passage" : "passages")\(visuals)")
                jobs[index].errorDetails = nil
                jobs[index].chunkCount = chunks
                try? FileManager.default.removeItem(at: markdownURL)
                if let metadataURL = jobs[index].metadataURL {
                    try? FileManager.default.removeItem(at: metadataURL)
                }
                if let bundleURL = jobs[index].visualArtifactBundleURL {
                    try? FileManager.default.removeItem(at: bundleURL)
                }
                jobs[index].artifactURL = nil
                jobs[index].metadataURL = nil
                jobs[index].visualArtifactBundleURL = nil
            } catch {
                jobs[index].errorDetails = (error as? PipelineError)?.diagnosticDetails
                    ?? error.localizedDescription
                update(index, stage: .readyForIndexing, detail: "Verification needs retrying: \(error.localizedDescription)")
            }
        }

        isRunning = false
        activeTask = nil
        let completed = transferred.filter { jobs[$0].stage == .completed }.count
        message = completed == 1
            ? "The PDF is available in Ragdoc."
            : "\(completed) PDFs are available in Ragdoc."
    }

    private func update(_ index: Int, stage: ImportStage, detail: String) {
        jobs[index].recordProgress(for: stage)
        jobs[index].stage = stage
        jobs[index].detail = detail
        jobs[index].stageStartedAt = Date()
    }

    private func removeJob(at index: Int) {
        removeTemporaryFiles(for: jobs[index])
        jobs.remove(at: index)
    }

    private func removeTemporaryFiles(for job: ImportJob) {
        guard !isIsolated else { return }
        for url in [job.artifactURL, job.metadataURL, job.visualArtifactBundleURL].compactMap({ $0 }) {
            try? FileManager.default.removeItem(at: url)
        }
    }

    private func updateMessageAfterRemoval() {
        message = jobs.isEmpty
            ? "Drop PDFs to begin."
            : "Item removed from the queue."
    }

    private static var defaultQueueStoreURL: URL {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return support.appendingPathComponent("Ragdrop", isDirectory: true)
            .appendingPathComponent("queue.json")
    }

    private static func loadJobs(from url: URL) -> [ImportJob] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode([ImportJob].self, from: data)) ?? []
    }

    private static func recoverInterruptedJob(_ stored: ImportJob) -> ImportJob {
        var job = stored
        let hasMarkdown = job.artifactURL.map {
            FileManager.default.isReadableFile(atPath: $0.path)
        } == true
        switch job.stage {
        case .checkingDuplicate, .converting:
            job.stage = .queued
            job.detail = "Processing interrupted · ready to retry"
            job.errorDetails = nil
        case .transferring, .indexing, .verifying:
            job.stage = hasMarkdown ? .readyForIndexing : .queued
            job.detail = hasMarkdown
                ? "Import interrupted · ready to resume"
                : "Processing interrupted · ready to retry"
            job.errorDetails = nil
        case .awaitingReview where !hasMarkdown,
             .readyForIndexing where !hasMarkdown:
            job.stage = .queued
            job.detail = "Temporary conversion missing · ready to retry"
            job.artifactURL = nil
            job.metadataURL = nil
            job.visualArtifactBundleURL = nil
            job.visualArtifactCount = 0
            job.errorDetails = nil
        default:
            break
        }
        job.stageStartedAt = Date()
        return job
    }

    private func persistJobs() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(jobs) else { return }
        let directory = queueStoreURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? data.write(to: queueStoreURL, options: .atomic)
    }
}
