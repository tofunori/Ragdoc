import Testing
@testable import Ragdrop

struct PipelineSafetyTests {
    @Test func rejectsShellSyntaxInRemoteConfiguration() {
        let configuration = PipelineConfiguration(
            converterPath: "/tmp/mineru.py",
            nasHost: "rorqual",
            remoteRoot: "/volume1/ragdoc'; touch /tmp/injected; echo '"
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
        #expect(!PipelineConfiguration.isLegacyConverterPath("/tmp/custom_converter.py"))
    }
}
