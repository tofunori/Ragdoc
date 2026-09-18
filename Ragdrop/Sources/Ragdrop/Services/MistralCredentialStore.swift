import Foundation
import Security

enum MistralCredentialStore {
    static let service = "com.tofunori.ragdrop.mistral"
    static let account = "api-key"

    static var isConfigured: Bool {
        if ProcessInfo.processInfo.environment["MISTRAL_API_KEY"]?.trimmedNonEmpty != nil {
            return true
        }
        let keyFile = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mistral_api_key")
        if let value = try? String(contentsOf: keyFile, encoding: .utf8),
           value.trimmedNonEmpty != nil {
            return true
        }
        return hasKeychainCredential
    }

    static var hasKeychainCredential: Bool {
        var result: CFTypeRef?
        var query = identityQuery
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        return status == errSecSuccess
    }

    static func save(_ value: String) throws {
        guard let credential = value.trimmedNonEmpty,
              let data = credential.data(using: .utf8) else {
            throw CredentialError.empty
        }
        SecItemDelete(identityQuery as CFDictionary)
        var attributes = identityQuery
        attributes[kSecValueData as String] = data
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        let status = SecItemAdd(attributes as CFDictionary, nil)
        guard status == errSecSuccess else { throw CredentialError.keychain(status) }
    }

    static func delete() throws {
        let status = SecItemDelete(identityQuery as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw CredentialError.keychain(status)
        }
    }

    private static var identityQuery: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
    }
}

enum CredentialError: LocalizedError {
    case empty
    case keychain(OSStatus)

    var errorDescription: String? {
        switch self {
        case .empty:
            "The Mistral key is empty."
        case .keychain(let status):
            "macOS Keychain rejected the key (code \(status))."
        }
    }
}

private extension String {
    var trimmedNonEmpty: String? {
        let trimmed = trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
