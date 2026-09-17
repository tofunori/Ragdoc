import Foundation

/// Counts actual queue states; elapsed work is not a duration estimate.
struct BatchSummary {
    let jobs: [ImportJob]
    func count(_ stages: ImportStage...) -> Int { jobs.filter { stages.contains($0.stage) }.count }
    var added: Int { count(.completed) }
    var review: Int { count(.awaitingReview) }
    var errors: Int { jobs.filter { $0.stage == .failed || $0.errorDetails != nil }.count }
    var active: Int { jobs.filter { $0.stage.isActive }.count }
}
