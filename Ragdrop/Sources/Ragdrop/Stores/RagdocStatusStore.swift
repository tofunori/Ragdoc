import Foundation
import Observation

@MainActor
@Observable
final class RagdocStatusStore {
    let isIsolated: Bool
    init(isIsolated: Bool = false) { self.isIsolated = isIsolated }
    var snapshot: RagdocStatusSnapshot?
    var isChecking = false
    var errorMessage: String?
    var lastChecked: Date?

    func refresh() async {
        guard !isIsolated else { return }
        guard !isChecking else { return }
        isChecking = true
        errorMessage = nil
        do {
            snapshot = try await RagdocStatusService(configuration: .current()).check()
            lastChecked = Date()
        } catch {
            snapshot = nil
            errorMessage = error.localizedDescription
        }
        isChecking = false
    }
}
