import Foundation
import Observation

@MainActor
@Observable
final class HistoryStore {
    let isIsolated: Bool
    init(isIsolated: Bool = false) { self.isIsolated = isIsolated }
    var documents: [HistoryDocument] = []
    var isLoading = false
    var errorMessage: String?
    var lastUpdated: Date?

    var recentDocuments: [HistoryDocument] {
        Array(documents.sorted {
            if ($0.indexedDate ?? "") != ($1.indexedDate ?? "") { return ($0.indexedDate ?? "") > ($1.indexedDate ?? "") }
            return $0.source < $1.source
        }.prefix(3))
    }

    func refresh() async {
        guard !isIsolated else { return }
        guard !isLoading else { return }
        isLoading = true
        errorMessage = nil
        do {
            documents = try await RagdocHistoryService(configuration: .current()).fetchDocuments()
            lastUpdated = Date()
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}
