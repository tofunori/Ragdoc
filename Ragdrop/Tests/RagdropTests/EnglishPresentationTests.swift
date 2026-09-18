import Foundation
import Testing
@testable import Ragdrop

struct EnglishPresentationTests {
    @Test func legacyQueueKeepsIdentifiersAndSourceTextWhileDisplayingEnglish() throws {
        let fixture = #"{"id":"11111111-1111-1111-1111-111111111111","fileURL":"file:///tmp/Albedo.pdf","stage":"completed","detail":"27 passages indexés · 22 éléments visuels","stageStartedAt":0,"visualArtifactCount":22,"progressHighWater":1,"metadata":{"title":"Albédo de la neige","authors":["Auteur original"]}}"#
        let job = try JSONDecoder().decode(ImportJob.self, from: Data(fixture.utf8))
        #expect(job.stage == .completed)
        #expect(job.displayDetail == "27 indexed passages · 22 visual items")
        #expect(job.detail == "27 passages indexés · 22 éléments visuels")
        #expect(job.metadata?.title == "Albédo de la neige")
        let encoded = try JSONEncoder().encode(job)
        let restored = try JSONDecoder().decode(ImportJob.self, from: encoded)
        #expect(restored == job)
        #expect(RagdropAppearance(rawValue: "dark") == .dark)
        #expect(ImportStage(rawValue: "awaitingReview")?.title == "Review needed")
    }
    @Test func originalDiagnosticsAreNotRewritten() {
        var job = ImportJob(fileURL: URL(fileURLWithPath: "/tmp/source.pdf"))
        job.detail = "Erreur du fournisseur : document illisible"
        #expect(job.displayDetail == job.detail)
        job.detail = "1 passages indexés · 1 éléments visuels"
        #expect(job.displayDetail == "1 indexed passage · 1 visual item")
    }
    @Test func queueCountsUseEnglishSingularAndPlural() {
        #expect(RagdropText.pdfCount(0) == "0 PDFs")
        #expect(RagdropText.pdfCount(1) == "1 PDF")
        #expect(RagdropText.pdfCount(2) == "2 PDFs")
        #expect(RagdropText.addedCount(0, total: 2) == "0 / 2 articles added")
        #expect(RagdropText.addedCount(1, total: 2) == "1 / 2 article added")
        #expect(RagdropText.addedCount(2, total: 2) == "2 / 2 articles added")
    }
}
