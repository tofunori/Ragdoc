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
    var message = "Glissez des PDF pour commencer."
    @ObservationIgnored private var activeTask: Task<Void, Never>?
    @ObservationIgnored private let queueStoreURL: URL

    init(queueStoreURL: URL = ImportStore.defaultQueueStoreURL, isIsolated: Bool = false) {
        self.isIsolated = isIsolated
        self.queueStoreURL = queueStoreURL
        jobs = Self.loadJobs(from: queueStoreURL).map(Self.recoverInterruptedJob)
        if !jobs.isEmpty {
            message = "Lot précédent récupéré. Vous pouvez reprendre le traitement."
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
                ? "Un PDF est prêt."
                : "\(additions.count) PDF sont prêts."
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
                ? "Un PDF Zotero est prêt."
                : "\(additions.count) PDF Zotero sont prêts."
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
        update(index, stage: .queued, detail: "Prêt à être relancé")
        isRunning = true
        activeTask = Task { await prepareQueue(only: Set([id])) }
    }

    func cancelConversion() {
        guard isRunning, jobs.contains(where: {
            $0.stage == .checkingDuplicate || $0.stage == .converting
        }) else { return }
        message = "Annulation de la conversion…"
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
        update(index, stage: .readyForIndexing, detail: "Prêt à être envoyé au NAS")
        message = "Markdown approuvé. Vous pouvez l’ajouter à Ragdoc."
    }

    func reject(_ id: UUID) {
        guard !isRunning, let index = jobs.firstIndex(where: { $0.id == id }),
              jobs[index].stage == .awaitingReview else { return }
        removeTemporaryFiles(for: jobs[index])
        jobs[index].artifactURL = nil
        jobs[index].metadataURL = nil
        jobs[index].visualArtifactBundleURL = nil
        update(index, stage: .rejected, detail: "Markdown écarté")
        message = "Le document a été écarté."
    }

    func approveAll() {
        guard !isRunning else { return }
        for index in jobs.indices where jobs[index].stage == .awaitingReview {
            update(index, stage: .readyForIndexing, detail: "Prêt à être envoyé au NAS")
        }
        message = "Les Markdown sont approuvés. Vous pouvez lancer l’ajout."
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
                update(index, stage: .checkingDuplicate, detail: "Comparaison de l’empreinte du PDF")
                message = "Recherche de doublon pour \(jobs[index].displayName)…"
                let duplicate = try await pipeline.duplicateFilename(for: jobs[index].fileURL)
                jobs[index].fingerprint = duplicate.fingerprint
                if fingerprintsSeen.contains(duplicate.fingerprint) {
                    update(index, stage: .duplicate, detail: "Même PDF déjà présent dans ce lot")
                    continue
                }
                if let filename = duplicate.filename {
                    update(index, stage: .duplicate, detail: "Déjà présent : \(filename)")
                    continue
                }

                jobs[index].converterName = configuration.converterKind.commandLabel
                update(index, stage: .converting, detail: "Analyse du PDF · \(configuration.converterKind.commandLabel)")
                message = "Conversion de \(jobs[index].displayName)…"
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
                    ? "1 tableau ou figure détecté"
                    : "\(artifact.artifactCount) tableaux et figures détectés"
                update(index, stage: .awaitingReview, detail: "Markdown prêt · \(visuals)")
            } catch is CancellationError {
                jobs[index].errorDetails = "La conversion a été annulée par l’utilisateur."
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
            message = "Conversion annulée. Le PDF peut être relancé."
            return
        }
        let ready = jobs.filter { $0.stage == .awaitingReview }.count
        message = ready == 1
            ? "Le Markdown est prêt. Vérifiez-le avant l’ajout."
            : "\(ready) Markdown sont prêts à vérifier."
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
                jobs[index].errorDetails = "Le fichier Markdown temporaire attendu par Ragdrop n’existe plus."
                update(index, stage: .failed, detail: "Le Markdown temporaire est introuvable.")
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
            update(index, stage: .transferring, detail: "Copie atomique du Markdown")
            do {
                try await pipeline.transfer(artifact)
                transferred.append(index)
                update(index, stage: .readyForIndexing, detail: "Transféré, en attente de l’indexation du lot")
            } catch {
                jobs[index].errorDetails = (error as? PipelineError)?.diagnosticDetails
                    ?? error.localizedDescription
                update(index, stage: .readyForIndexing, detail: "Transfert à reprendre : \(error.localizedDescription)")
            }
        }

        guard !transferred.isEmpty else {
            isRunning = false
            message = "Aucun Markdown n’a pu être transféré."
            return
        }

        for index in transferred {
            update(index, stage: .indexing, detail: "Création des embeddings")
        }
        message = "Indexation du lot dans Ragdoc…"

        do {
            let sources = transferred.compactMap { jobs[$0].artifactURL?.lastPathComponent }
            try await pipeline.indexAll(sources: sources)
        } catch {
            let diagnostic = (error as? PipelineError)?.diagnosticDetails
                ?? error.localizedDescription
            for index in transferred {
                jobs[index].errorDetails = diagnostic
                update(index, stage: .readyForIndexing, detail: "Indexation à reprendre : \(error.localizedDescription)")
            }
            isRunning = false
            message = "L’indexation a échoué."
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
            update(index, stage: .verifying, detail: "Comptage des passages dans la base canonique")
            do {
                let chunks = try await pipeline.verify(artifact)
                let visuals = artifact.artifactCount > 0 ? " · \(artifact.artifactCount) éléments visuels" : ""
                update(index, stage: .completed, detail: "\(chunks) passages indexés\(visuals)")
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
                update(index, stage: .readyForIndexing, detail: "Vérification à reprendre : \(error.localizedDescription)")
            }
        }

        isRunning = false
        activeTask = nil
        let completed = transferred.filter { jobs[$0].stage == .completed }.count
        message = completed == 1
            ? "Le PDF est disponible dans Ragdoc."
            : "\(completed) PDF sont disponibles dans Ragdoc."
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
            ? "Glissez des PDF pour commencer."
            : "Élément retiré de la file."
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
            job.detail = "Traitement interrompu · prêt à être relancé"
            job.errorDetails = nil
        case .transferring, .indexing, .verifying:
            job.stage = hasMarkdown ? .readyForIndexing : .queued
            job.detail = hasMarkdown
                ? "Ajout interrompu · prêt à être repris"
                : "Traitement interrompu · prêt à être relancé"
            job.errorDetails = nil
        case .awaitingReview where !hasMarkdown,
             .readyForIndexing where !hasMarkdown:
            job.stage = .queued
            job.detail = "Conversion temporaire absente · prêt à être relancé"
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
