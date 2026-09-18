import SwiftUI

struct DropZoneView: View {
    let isTargeted: Bool
    var compact = false
    let chooseFiles: () -> Void
    let chooseZotero: () -> Void
    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 20) { introduction; Spacer(minLength: 12); actions }
            VStack(alignment: .leading, spacing: 18) { introduction; actions }
        }
        .padding(22)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(isTargeted ? RagdropTheme.signal.opacity(0.10) : RagdropTheme.panel, in: .rect(cornerRadius: 5))
        .overlay { RoundedRectangle(cornerRadius: 5).strokeBorder(isTargeted ? RagdropTheme.link : RagdropTheme.line, lineWidth: isTargeted ? 2 : 1) }
        .contentShape(.rect)
    }
    private var introduction: some View {
        HStack(spacing: 16) {
            Image(systemName: "doc.badge.plus").font(.system(size: 26, weight: .light)).foregroundStyle(RagdropTheme.text)
            VStack(alignment: .leading, spacing: 5) {
                Text(isTargeted ? "Drop PDFs here" : "Add articles").font(.system(size: 15, weight: .semibold))
                Text("Local PDFs or Zotero attachments").font(.callout).foregroundStyle(RagdropTheme.secondary)
            }
        }
    }
    private var actions: some View {
        HStack(spacing: 10) {
            Button("Choose PDFs", systemImage: "doc.badge.plus", action: chooseFiles).buttonStyle(RagdropPrimaryButtonStyle())
            Button("From Zotero", systemImage: "books.vertical", action: chooseZotero).buttonStyle(RagdropSecondaryButtonStyle())
        }.controlSize(.large).fixedSize()
    }
}
