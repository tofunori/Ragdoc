import Foundation
import Testing
@testable import Ragdrop

struct PipelineSafetyTests {
    @Test func rejectsShellSyntaxInRemoteConfiguration() {
        let configuration = PipelineConfiguration(
            converterPath: "/tmp/mineru.py",
            nasHost: "ragdoc-server",
            remoteRoot: "/srv/ragdoc'; touch /tmp/injected; echo '"
        )
        #expect(throws: PipelineError.self) {
            try configuration.validate()
        }
    }

    @Test func onlyRetainedDocumentsReserveTheirFingerprint() {
        #expect(ImportStage.awaitingReview.ownsFingerprint)
        #expect(ImportStage.readyForIndexing.ownsFingerprint)
        #expect(ImportStage.completed.ownsFingerprint)
        #expect(!ImportStage.failed.ownsFingerprint)
        #expect(!ImportStage.duplicate.ownsFingerprint)
        #expect(!ImportStage.rejected.ownsFingerprint)
    }

    @Test func migratesOnlyTheLegacyMinerUConverter() {
        #expect(PipelineConfiguration.isLegacyConverterPath(
            "/Users/example/.codex/skills/mineru-pdf/mineru_convert.py"
        ))
        #expect(PipelineConfiguration.isLegacyConverterPath(
            "/Applications/Ragdrop.app/Contents/Resources/ragdrop_mineru_convert.py"
        ))
        #expect(!PipelineConfiguration.isLegacyConverterPath("/tmp/custom_converter.py"))
    }

    @Test func recognizesMistralMinerUAndCustomConverters() {
        #expect(PipelineConfiguration.converterKind(for: "/tmp/ragdrop_mistral_convert.py") == .mistral)
        #expect(PipelineConfiguration.converterKind(for: "/tmp/ragdrop_mineru_convert.py") == .mineru)
        #expect(PipelineConfiguration.converterKind(for: "/tmp/parser.py") == .custom)
        #expect(FileManager.default.fileExists(atPath: PipelineConfiguration.defaultMistralConverterPath))
    }

    @Test func providerErrorsNameTheActiveService() {
        let size = PipelineError.oversizedPDF(provider: "Mistral OCR", limitMB: 512)
        let output = PipelineError.missingOutput("Mistral OCR")

        #expect(size.localizedDescription.contains("512 Mo"))
        #expect(output.localizedDescription.contains("Mistral OCR"))
    }

    @Test func processFailureShowsTheCauseAndRetainsDiagnostics() {
        let details = "Traceback line\nConnectionError: broken pipe\n"
        let error = PipelineError.processFailed(command: "MinerU", details: details)

        #expect(error.localizedDescription == "MinerU a échoué. ConnectionError: broken pipe")
        #expect(error.diagnosticDetails == details)
    }

    @Test func timeoutExplainsThatThePDFCanBeRetried() {
        let error = PipelineError.timedOut(command: "MinerU", minutes: 30)

        #expect(error.localizedDescription.contains("30 minutes"))
        #expect(error.localizedDescription.contains("relancer"))
    }

    @MainActor
    @Test func removingAJobClearsItsTemporaryArtifactsOnly() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("ragdrop-remove-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let source = folder.appendingPathComponent("paper.pdf")
        let markdown = folder.appendingPathComponent("paper.md")
        let metadata = folder.appendingPathComponent("paper.metadata.json")
        let bundle = folder.appendingPathComponent("paper.ragdoc-artifacts", isDirectory: true)
        try Data("pdf".utf8).write(to: source)
        try Data("markdown".utf8).write(to: markdown)
        try Data("{}".utf8).write(to: metadata)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)

        var job = ImportJob(fileURL: source)
        job.stage = .awaitingReview
        job.artifactURL = markdown
        job.metadataURL = metadata
        job.visualArtifactBundleURL = bundle
        let store = ImportStore(queueStoreURL: folder.appendingPathComponent("queue.json"))
        store.jobs = [job]

        store.remove(job.id)

        #expect(store.jobs.isEmpty)
        #expect(FileManager.default.fileExists(atPath: source.path))
        #expect(!FileManager.default.fileExists(atPath: markdown.path))
        #expect(!FileManager.default.fileExists(atPath: metadata.path))
        #expect(!FileManager.default.fileExists(atPath: bundle.path))
    }

    @Test func progressDoesNotMoveBackwardAfterAnIndexingFailure() {
        var job = ImportJob(fileURL: URL(fileURLWithPath: "/tmp/paper.pdf"))
        job.recordProgress(for: .transferring)
        job.recordProgress(for: .indexing)
        job.stage = .indexing
        job.stage = .readyForIndexing

        // The job is retryable, but the UI remembers that transfer and
        // indexation were already reached instead of jumping back to 35%.
        #expect(job.overallProgress == 0.66)
    }

    @MainActor
    @Test func queueRestoresConvertedWorkAndResetsInterruptedConversion() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("ragdrop-queue-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let queueURL = folder.appendingPathComponent("queue.json")
        let source = folder.appendingPathComponent("paper.pdf")
        let markdown = folder.appendingPathComponent("paper.md")
        try Data("pdf".utf8).write(to: source)
        try Data("markdown".utf8).write(to: markdown)

        let writer = ImportStore(queueStoreURL: queueURL)
        var converted = ImportJob(fileURL: source)
        converted.stage = .awaitingReview
        converted.detail = "Markdown prêt"
        converted.artifactURL = markdown
        var interrupted = ImportJob(fileURL: source.appendingPathExtension("second"))
        interrupted.stage = .converting
        writer.jobs = [converted, interrupted]

        let restored = ImportStore(queueStoreURL: queueURL)

        #expect(restored.jobs.count == 2)
        #expect(restored.jobs[0].stage == .awaitingReview)
        #expect(restored.jobs[0].artifactURL == markdown)
        #expect(restored.jobs[1].stage == .queued)
        #expect(restored.jobs[1].detail.contains("interrompu"))
    }
}
