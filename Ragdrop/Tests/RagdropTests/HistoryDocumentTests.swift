import Foundation
import Testing
@testable import Ragdrop

struct HistoryDocumentTests {
    @Test func decodesNASHistoryRows() throws {
        let data = Data(#"[{"source":"paper_a.md","title":"Paper A","chunks":12,"indexedDate":"2026-09-15T20:10:11","doi":"10.1234/example"}]"#.utf8)
        let document = try #require(JSONDecoder().decode([HistoryDocument].self, from: data).first)
        #expect(document.displayTitle == "Paper A")
        #expect(document.displayDate == "2026-09-15 20:10")
        #expect(document.chunks == 12)
    }
}
