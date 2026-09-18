import Foundation

/// Five workflow stages, never a measurement of time or work completed.
enum BatchPhase: Int, CaseIterable {
    case conversion, review, transfer, indexing, verification
    var title: String {
        switch self {
        case .conversion: "Conversion"
        case .review: "Review"
        case .transfer: "Transfer"
        case .indexing: "Indexing"
        case .verification: "Verification"
        }
    }
}

struct BatchPresentation {
    let jobs: [ImportJob]
    var summary: BatchSummary { BatchSummary(jobs: jobs) }
    var focus: ImportJob? {
        // Indexing is collective; during sequential verification other rows may still say indexing.
        jobs.first { $0.stage == .verifying }
            ?? jobs.first { $0.stage.isActive }
            ?? jobs.first { $0.stage == .failed || $0.errorDetails != nil }
            ?? jobs.first { $0.stage == .awaitingReview }
            ?? jobs.first { $0.stage == .readyForIndexing }
            ?? jobs.first { $0.stage == .queued }
            ?? jobs.first
    }
    var hasFailure: Bool { summary.errors > 0 }
    var allCompleted: Bool { !jobs.isEmpty && jobs.allSatisfy { $0.stage == .completed } }
    var terminal: Bool { !jobs.isEmpty && jobs.allSatisfy { [.completed, .duplicate, .rejected].contains($0.stage) && $0.errorDetails == nil } }
    var phase: BatchPhase? {
        guard let focus, focus.errorDetails == nil, !terminal else { return nil }
        switch focus.stage {
        case .queued, .checkingDuplicate, .converting: return .conversion
        case .awaitingReview: return .review
        case .readyForIndexing, .transferring: return .transfer
        case .indexing: return .indexing
        case .verifying: return .verification
        default: return nil
        }
    }
    var isWorking: Bool { focus?.stage.isActive == true && focus?.errorDetails == nil }
    var headline: String {
        guard let focus else { return "No articles waiting" }
        if !isWorking && hasFailure { return "Processing needs attention" }
        if terminal {
            if allCompleted { return jobs.count == 1 ? "Article added" : "Articles added" }
            if jobs.allSatisfy({ $0.stage == .duplicate }) { return jobs.count == 1 ? "Duplicate detected" : "Duplicates detected" }
            if jobs.allSatisfy({ $0.stage == .rejected }) { return jobs.count == 1 ? "Article rejected" : "Articles rejected" }
            return "Batch finished"
        }
        switch focus.stage {
        case .queued: return "Ready to start"
        case .checkingDuplicate: return "Checking for duplicates"
        case .converting: return "Converting"
        case .awaitingReview: return "Review needed"
        case .readyForIndexing: return "Ready to add"
        case .transferring: return "Transferring"
        case .indexing: return "Indexing"
        case .verifying: return "Verifying"
        default: return "Processing needs attention"
        }
    }
    var subtitle: String {
        if let phase {
            let step = "Step \(phase.rawValue + 1) of 5"
            switch focus?.stage {
            case .queued, .readyForIndexing: return "\(step) · waiting to start"
            case .awaitingReview: return "\(step) · your review is required"
            default: return step
            }
        }
        if hasFailure { return "Check the errors before retrying." }
        if allCompleted { return "Passages found in Ragdoc." }
        if terminal { return "No processing remaining." }
        return "Add a PDF to begin."
    }
    /// Completed segments refer only to the displayed article, never the whole mixed batch.
    func hasPassed(_ candidate: BatchPhase) -> Bool {
        if allCompleted { return true }
        guard let phase else { return false }
        return candidate.rawValue < phase.rawValue
    }
    var context: String? {
        var parts: [String] = []
        if summary.review > 0 && focus?.stage != .awaitingReview {
            parts.append("\(summary.review) to review")
        }
        if hasFailure { parts.append("\(summary.errors) to retry") }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}
