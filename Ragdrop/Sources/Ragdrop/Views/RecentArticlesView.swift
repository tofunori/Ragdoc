import SwiftUI

struct RecentArticlesView: View {
    @Bindable var store: HistoryStore
    let onLibrary: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Recent articles").font(.system(size: 20, weight: .semibold))
                Spacer()
                Button("View library →", action: onLibrary).buttonStyle(.plain).foregroundStyle(RagdropTheme.link)
            }
            if store.isLoading && store.documents.isEmpty {
                HStack(spacing: 10) { ProgressView().controlSize(.small); Text("Reading library…") }
                    .foregroundStyle(RagdropTheme.secondary).padding(.vertical, 20)
            } else if let error = store.errorMessage, store.documents.isEmpty {
                HStack {
                    Label("Recent articles unavailable", systemImage: "wifi.exclamationmark")
                    Spacer()
                    Button("Try again") { Task { await store.refresh() } }.disabled(store.isLoading)
                }.foregroundStyle(RagdropTheme.secondary)
                Text(error).font(.caption).foregroundStyle(RagdropTheme.secondary).lineLimit(2)
            } else if store.documents.isEmpty {
                Text("Your articles will appear here after being added to Ragdoc.")
                    .foregroundStyle(RagdropTheme.secondary).padding(.vertical, 14)
            } else {
                VStack(spacing: 0) {
                    HStack {
                        Text("ARTICLE").frame(maxWidth: .infinity, alignment: .leading)
                        Text("PASSAGES").frame(width: 80, alignment: .leading)
                        Text("INDEXED ON").frame(width: 145, alignment: .leading)
                    }.font(.caption).foregroundStyle(RagdropTheme.secondary).padding(.vertical, 10)
                    Rectangle().fill(RagdropTheme.line).frame(height: 1)
                    ForEach(store.recentDocuments) { document in
                        HStack(spacing: 14) {
                            Image(systemName: "doc.text").font(.system(size: 18, weight: .light)).frame(width: 24)
                            Text(document.displayTitle).font(.body).lineLimit(2).help(document.displayTitle)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Text(document.chunks.formatted()).monospacedDigit().frame(width: 80, alignment: .leading)
                            Text(document.displayDate).monospacedDigit().frame(width: 145, alignment: .leading)
                        }.font(.callout).padding(.vertical, 15)
                        Rectangle().fill(RagdropTheme.line).frame(height: 1)
                    }
                }
                if store.errorMessage != nil {
                    Text("Showing the last snapshot · refresh unavailable").font(.caption).foregroundStyle(RagdropTheme.warning)
                }
            }
        }
        .task { if store.lastUpdated == nil && store.errorMessage == nil { await store.refresh() } }
    }
}
