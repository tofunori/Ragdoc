import Foundation
import Observation

@MainActor
@Observable
final class HistoryStore {
    var documents: [HistoryDocument] = []
    var isLoading = false
    var errorMessage: String?
    var lastUpdated: Date?

    func refresh() async {
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
