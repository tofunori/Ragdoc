import Foundation

/// English presentation text; persisted identifiers and source content are unchanged.
enum RagdropText {
    static func pdfCount(_ count: Int) -> String {
        "\(count) \(count == 1 ? "PDF" : "PDFs")"
    }

    static func addedCount(_ count: Int, total: Int) -> String {
        "\(count) / \(total) \(count == 1 ? "article added" : "articles added")"
    }
}

extension ImportJob {
    /// Present known legacy app messages without rewriting the saved queue or raw diagnostics.
    var displayDetail: String {
        let messages = [
            "Prêt à être ajouté": "Ready to add",
            "Prêt à être relancé": "Ready to retry",
            "Prêt à être envoyé au NAS": "Ready to send to the server",
            "Markdown écarté": "Markdown rejected",
            "Même PDF déjà présent dans ce lot": "The same PDF is already in this batch",
            "Traitement interrompu · prêt à être relancé": "Processing interrupted · ready to retry",
            "Ajout interrompu · prêt à être repris": "Import interrupted · ready to resume"
        ]
        if let translated = messages[detail] { return translated }
        // Counts are app-generated; filenames, article titles and arbitrary diagnostics are not translated.
        let pattern = #"^(\d+) passages indexés(?: · (\d+) éléments visuels)?$"#
        if let regex = try? NSRegularExpression(pattern: pattern),
           let match = regex.firstMatch(in: detail, range: NSRange(detail.startIndex..., in: detail)),
           let countRange = Range(match.range(at: 1), in: detail),
           let count = Int(detail[countRange]) {
            var result = "\(count) indexed \(count == 1 ? "passage" : "passages")"
            if let visualRange = Range(match.range(at: 2), in: detail), let visuals = Int(detail[visualRange]) {
                result += " · \(visuals) visual \(visuals == 1 ? "item" : "items")"
            }
            return result
        }
        return detail
    }
}
