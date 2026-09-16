import Foundation
import Observation

@MainActor
@Observable
final class ImportStore {
    var jobs: [ImportJob] = []
    var isRunning = false
    var message = "Glissez des PDF pour commencer."

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
        let existing = Set(jobs.map { $0.fileURL.standardizedFileURL })
        let additions = documents
            .filter { !existing.contains($0.fileURL.standardizedFileURL) }
            .map { ImportJob(fileURL: $0.fileURL, metadata: $0.importMetadata) }
        jobs.append(contentsOf: additions)
        if !additions.isEmpty {
            message = additions.count == 1
                ? "Un PDF Zotero est prêt."
                : "\(additions.count) PDF Zotero sont prêts."
        }
    }

    func removeJobs(at offsets: IndexSet) {
        guard !isRunning else { return }
        jobs.remove(atOffsets: offsets)
        if jobs.isEmpty { message = "Glissez des PDF pour commencer." }
    }

    func clearCompleted() {
        guard !isRunning else { return }
        jobs.removeAll { [.completed, .duplicate, .rejected].contains($0.stage) }
    }

    func start() {
        guard !isRunning else { return }
        isRunning = true
        if jobs.contains(where: { $0.stage == .readyForIndexing }) {
            Task { await importApproved() }
        } else if jobs.contains(where: { $0.stage == .queued || $0.stage == .failed }) {
            Task { await prepareQueue() }
        } else {
            isRunning = false
        }
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
        if let artifactURL = jobs[index].artifactURL { try? FileManager.default.removeItem(at: artifactURL) }
        if let metadataURL = jobs[index].metadataURL { try? FileManager.default.removeItem(at: metadataURL) }
        if let bundleURL = jobs[index].visualArtifactBundleURL { try? FileManager.default.removeItem(at: bundleURL) }
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

    private func prepareQueue() async {
        let pipeline = ImportPipeline(configuration: .current())
        let candidates = jobs.indices.filter { jobs[$0].stage == .queued || jobs[$0].stage == .failed }
        let candidateSet = Set(candidates)
        var fingerprintsSeen = Set(
            jobs.indices
                .filter { !candidateSet.contains($0) && jobs[$0].stage.ownsFingerprint }
                .compactMap { jobs[$0].fingerprint }
        )

        for index in candidates {
            do {
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

                update(index, stage: .converting, detail: "Analyse du PDF")
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
            } catch {
                update(index, stage: .failed, detail: error.localizedDescription)
            }
        }

        isRunning = false
        let ready = jobs.filter { $0.stage == .awaitingReview }.count
        message = ready == 1
            ? "Le Markdown est prêt. Vérifiez-le avant l’ajout."
            : "\(ready) Markdown sont prêts à vérifier."
    }

    private func importApproved() async {
        let pipeline = ImportPipeline(configuration: .current())
        let approved = jobs.indices.filter { jobs[$0].stage == .readyForIndexing }
        var transferred: [Int] = []

        for index in approved {
            guard let markdownURL = jobs[index].artifactURL else {
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
            for index in transferred {
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
            update(index, stage: .verifying, detail: "Lecture dans la base canonique")
            do {
                let chunks = try await pipeline.verify(artifact)
                let visuals = artifact.artifactCount > 0 ? " · \(artifact.artifactCount) éléments visuels" : ""
                update(index, stage: .completed, detail: "\(chunks) passages indexés\(visuals)")
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
                update(index, stage: .readyForIndexing, detail: "Vérification à reprendre : \(error.localizedDescription)")
            }
        }

        isRunning = false
        let completed = transferred.filter { jobs[$0].stage == .completed }.count
        message = completed == 1
            ? "Le PDF est disponible dans Ragdoc."
            : "\(completed) PDF sont disponibles dans Ragdoc."
    }

    private func update(_ index: Int, stage: ImportStage, detail: String) {
        jobs[index].stage = stage
        jobs[index].detail = detail
        jobs[index].stageStartedAt = Date()
    }
}
