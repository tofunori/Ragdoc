import Foundation

struct HistoryDocument: Codable, Identifiable, Sendable {
    let source: String
    let title: String
    let chunks: Int
    let indexedDate: String?
    let doi: String?

    var id: String { source }

    var displayTitle: String {
        let cleaned = title.trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty ? source.replacingOccurrences(of: ".md", with: "") : cleaned
    }

    var displayDate: String {
        guard let indexedDate, !indexedDate.isEmpty else { return "—" }
        return indexedDate
            .replacingOccurrences(of: "T", with: " ")
            .prefix(16)
            .description
    }
}
