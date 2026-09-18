import Foundation
import Testing
@testable import Ragdrop

struct BatchPresentationTests {
    private func job(_ stage: ImportStage, error: String? = nil) -> ImportJob {
        var job = ImportJob(fileURL: URL(fileURLWithPath: "/tmp/test.pdf"))
        job.stage = stage; job.errorDetails = error
        return job
    }
    @Test func onlyControlledAdditionsCountAsAdded() {
        let p = BatchPresentation(jobs: [job(.completed), job(.completed), job(.duplicate), job(.rejected)])
        #expect(p.summary.added == 2 && p.jobs.count == 4)
        #expect(p.terminal && !p.allCompleted && p.phase == nil)
        #expect(BatchPhase.allCases.allSatisfy { !p.hasPassed($0) })
    }
    @Test func activeConversionDoesNotHideReviewOrErrors() {
        let p = BatchPresentation(jobs: [job(.awaitingReview), job(.failed), job(.converting)])
        #expect(p.focus?.stage == .converting && p.phase == .conversion && p.isWorking)
        #expect(p.context == "1 to review · 1 to retry")
        #expect(!p.hasPassed(.conversion))
    }
    @Test func humanReviewAndApprovedWaitDoNotAnimate() {
        let review = BatchPresentation(jobs: [job(.awaitingReview)])
        #expect(review.phase == .review && !review.isWorking)
        #expect(review.hasPassed(.conversion) && !review.hasPassed(.review))
        let ready = BatchPresentation(jobs: [job(.readyForIndexing)])
        #expect(ready.phase == .transfer && !ready.isWorking && !ready.hasPassed(.transfer))
    }
    @Test func recoverableErrorsOverrideHistoricalProgress() {
        var retry = job(.readyForIndexing, error: "Indexing failed")
        retry.progressHighWater = 0.88
        let p = BatchPresentation(jobs: [job(.completed), retry])
        #expect(p.hasFailure && !p.terminal && p.phase == nil && !p.isWorking)
        #expect(p.headline == "Processing needs attention")
        #expect(BatchPhase.allCases.allSatisfy { !p.hasPassed($0) })
    }
    @Test func sequentialVerificationTakesPrecedenceOverStaleIndexingRows() {
        let p = BatchPresentation(jobs: [job(.indexing), job(.verifying), job(.indexing)])
        #expect(p.phase == .verification && p.focus?.stage == .verifying)
        #expect(p.summary.added == 0)
    }
    @Test func terminalAndEmptyStatesRemainHonest() {
        for stage in [ImportStage.duplicate, .rejected] {
            let p = BatchPresentation(jobs: [job(stage)])
            #expect(p.terminal && p.summary.added == 0 && p.phase == nil && !p.isWorking)
        }
        #expect(BatchPresentation(jobs: [job(.duplicate)]).headline == "Duplicate detected")
        let done = BatchPresentation(jobs: [job(.completed)])
        #expect(done.allCompleted && BatchPhase.allCases.allSatisfy { done.hasPassed($0) })
        let empty = BatchPresentation(jobs: [])
        #expect(empty.focus == nil && empty.phase == nil && !empty.terminal && !empty.isWorking)
    }
}
