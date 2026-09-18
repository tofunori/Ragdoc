import SwiftUI

enum WorkspaceSection: String, CaseIterable, Identifiable {
    case home, review, library, status, settings
    var id: Self { self }
    var title: String {
        switch self {
        case .home: "Home"
        case .review: "To review"
        case .library: "Library"
        case .status: "Ragdoc · NAS"
        case .settings: "Settings"
        }
    }
    var symbol: String {
        switch self {
        case .home: "house"
        case .review: "checkmark.circle"
        case .library: "books.vertical"
        case .status: "externaldrive"
        case .settings: "gearshape"
        }
    }
}

struct WorkspaceView: View {
    @Bindable var store: ImportStore
    @Bindable var history: HistoryStore
    @Bindable var status: RagdocStatusStore
    @Bindable var monitor: ZoteroMonitorStore
    @Binding var section: WorkspaceSection
    @State private var libraryQuery = ""

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 12) {
                    RagdropBrandIcon().frame(width: 26, height: 26)
                    Text("Ragdrop").font(.system(size: 21, weight: .semibold)).tracking(-0.3)
                }.padding(.horizontal, 20).padding(.top, 64).padding(.bottom, 32)
                navigationButton(.home)
                navigationButton(.review)
                if !monitor.pendingDocuments.isEmpty {
                    Button { monitor.showingNewArticles = true } label: {
                        Label("Zotero · \(monitor.pendingDocuments.count) new", systemImage: "tray.and.arrow.down")
                            .font(.callout).foregroundStyle(RagdropTheme.link).padding(14)
                    }.buttonStyle(.plain).padding(.horizontal, 12)
                }
                navigationButton(.library)
                Spacer()
                Divider().padding(.horizontal, 20).padding(.bottom, 12)
                navigationButton(.status)
                navigationButton(.settings)
            }
            .padding(.bottom, 14)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(RagdropTheme.sidebar)
            .frame(width: 220)
            Divider()
            VStack(spacing: 0) {
                workspaceHeader
                Divider()
                if monitor.isEnabled && (!monitor.pendingDocuments.isEmpty || monitor.errorMessage != nil) {
                    ZoteroNotificationView(monitor: monitor).padding(.horizontal, RagdropTheme.pagePadding).padding(.top, 16)
                }
                Group {
                switch section {
                case .home, .review:
                    ContentView(store: store, history: history, reviewOnly: section == .review,
                                onLibrary: { section = .library })
                case .library: HistoryView(store: history, query: $libraryQuery)
                case .status: RagdocStatusView(store: status)
                case .settings: SettingsView(isIsolated: store.isIsolated, monitor: monitor)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(RagdropTheme.canvas)
            }
        }
        .onAppear { monitor.start(queue: store) }
        .onChange(of: monitor.isEnabled) { _, _ in monitor.start(queue: store) }
        .onChange(of: store.jobs) { _, jobs in monitor.reconcileQueue(jobs) }
        .sheet(isPresented: $monitor.showingNewArticles) {
            ZoteroImportView(store: monitor.selectionStore(jobs: store.jobs)) { documents in
                store.addZoteroDocuments(documents)
                monitor.dismiss(Set(documents.map(\.attachmentKey)))
            }
        }
        .onChange(of: store.jobs.filter { $0.stage == .completed }.count) { oldValue, newValue in
            if newValue > oldValue { Task { await history.refresh() } }
        }
        .ignoresSafeArea(.container, edges: .top)
        .ragdropSurface()
        .frame(minWidth: 1000, minHeight: 760)
    }

    private var workspaceHeader: some View {
        HStack(spacing: 10) {
            Text("Workspace").foregroundStyle(RagdropTheme.secondary)
            Spacer(minLength: 14)
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass").foregroundStyle(RagdropTheme.secondary)
                TextField("", text: $libraryQuery)
                    .overlay(alignment: .leading) {
                        if libraryQuery.isEmpty {
                            Text("Search your articles…")
                                .foregroundStyle(RagdropTheme.secondary)
                                .allowsHitTesting(false)
                                .accessibilityHidden(true)
                        }
                    }
                    .textFieldStyle(.plain)
                    .accessibilityLabel("Search the library")
                    .onChange(of: libraryQuery) { _, value in
                        if !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { section = .library }
                    }
            }.padding(.horizontal, 12).padding(.vertical, 8)
                .frame(maxWidth: 330)
                .background(RagdropTheme.raised.opacity(0.6), in: .rect(cornerRadius: 8))
                .overlay { RoundedRectangle(cornerRadius: 8).strokeBorder(RagdropTheme.line) }
        }
        .padding(.horizontal, RagdropTheme.pagePadding)
        .frame(height: 56)
        .background(RagdropTheme.canvas)
    }

    private func navigationButton(_ destination: WorkspaceSection) -> some View {
        Button { section = destination } label: {
            HStack(spacing: 12) {
                Image(systemName: destination.symbol).font(.system(size: 19, weight: .regular)).frame(width: 26)
                Text(destination.title).font(.system(size: 14, weight: section == destination ? .semibold : .regular))
                Spacer(minLength: 0)
                if destination == .review {
                    let count = store.jobs.filter { $0.stage == .awaitingReview }.count
                    if count > 0 { Text(count.formatted()).font(.caption.monospacedDigit()).padding(.horizontal, 7).padding(.vertical, 3).background(.white.opacity(0.09), in: Capsule()) }
                }
            }
            .foregroundStyle(section == destination ? RagdropTheme.text : RagdropTheme.secondary)
            .padding(.horizontal, 14).padding(.vertical, 13)
            .background(section == destination ? RagdropTheme.raised : .clear,
                        in: .rect(cornerRadius: 4))
            .overlay(alignment: .leading) {
                if section == destination { Rectangle().fill(RagdropTheme.accent).frame(width: 2).padding(.vertical, 5) }
            }
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(section == destination ? .isSelected : [])
        .accessibilityRemoveTraits(section == destination ? [] : .isSelected)
        .padding(.horizontal, 12).padding(.vertical, 2)
    }
}
