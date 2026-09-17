import Foundation
import Testing
@testable import Ragdrop

struct FilenameSanitizerTests {
    @Test func normalizesFinderNamesForRemoteUse() {
        let url = URL(fileURLWithPath: "/tmp/Énergie & albédo (2026).pdf")
        #expect(FilenameSanitizer.outputName(for: url) == "Energie_albedo_2026")
    }

    @Test func providesFallbackForPunctuationOnlyName() {
        let url = URL(fileURLWithPath: "/tmp/---.pdf")
        #expect(FilenameSanitizer.outputName(for: url) == "Article")
    }

    @Test func hashSuffixPreventsNormalizedNameCollisions() {
        let first = URL(fileURLWithPath: "/tmp/A&B.pdf")
        let second = URL(fileURLWithPath: "/tmp/A B.pdf")
        #expect(FilenameSanitizer.outputName(for: first, hashPrefix: "111111111111") !=
                FilenameSanitizer.outputName(for: second, hashPrefix: "222222222222"))
    }
}
