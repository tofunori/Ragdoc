import Foundation

enum FilenameSanitizer {
    static func outputName(for fileURL: URL, hashPrefix: String? = nil) -> String {
        let stem = fileURL.deletingPathExtension().lastPathComponent
            .folding(options: [.diacriticInsensitive, .widthInsensitive], locale: .current)
        let normalized = stem
            .replacingOccurrences(of: "[^A-Za-z0-9_-]+", with: "_", options: .regularExpression)
            .replacingOccurrences(of: "_+", with: "_", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "_-"))
        let fallback = normalized.isEmpty ? "Article" : normalized
        guard let hashPrefix, !hashPrefix.isEmpty else {
            return String(fallback.prefix(120))
        }
        return "\(String(fallback.prefix(105)))_\(hashPrefix)"
    }
}
