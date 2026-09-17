import Foundation

/// Five workflow stages, never a measurement of time or work completed.
enum BatchPhase: Int, CaseIterable {
    case conversion, review, transfer, indexing, verification
    var title: String {
        switch self {
        case .conversion: "Conversion"
        case .review: "Révision"
        case .transfer: "Transfert"
        case .indexing: "Indexation"
        case .verification: "Contrôle"
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
        guard let focus else { return "Aucun article en attente" }
        if !isWorking && hasFailure { return "Traitement à reprendre" }
        if terminal {
            if allCompleted { return jobs.count == 1 ? "Article ajouté" : "Articles ajoutés" }
            if jobs.allSatisfy({ $0.stage == .duplicate }) { return jobs.count == 1 ? "Doublon détecté" : "Doublons détectés" }
            if jobs.allSatisfy({ $0.stage == .rejected }) { return jobs.count == 1 ? "Article écarté" : "Articles écartés" }
            return "Lot terminé"
        }
        switch focus.stage {
        case .queued: return "Prêt à démarrer"
        case .checkingDuplicate: return "Recherche de doublon"
        case .converting: return "Conversion en cours"
        case .awaitingReview: return "Révision attendue"
        case .readyForIndexing: return "Prêt pour l’ajout"
        case .transferring: return "Transfert en cours"
        case .indexing: return "Indexation en cours"
        case .verifying: return "Contrôle en cours"
        default: return "Traitement à reprendre"
        }
    }
    var subtitle: String {
        if let phase {
            let step = "Étape \(phase.rawValue + 1) sur 5"
            switch focus?.stage {
            case .queued, .readyForIndexing: return "\(step) · en attente de lancement"
            case .awaitingReview: return "\(step) · votre validation est requise"
            default: return step
            }
        }
        if hasFailure { return "Consultez les erreurs avant de relancer." }
        if allCompleted { return "Passages retrouvés dans Ragdoc." }
        if terminal { return "Aucun traitement restant." }
        return "Ajoutez un PDF pour commencer."
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
            parts.append("\(summary.review) à réviser")
        }
        if hasFailure { parts.append("\(summary.errors) à reprendre") }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}
