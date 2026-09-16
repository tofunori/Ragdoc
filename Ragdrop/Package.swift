// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Ragdrop",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "Ragdrop", targets: ["Ragdrop"])
    ],
    targets: [
        .executableTarget(
            name: "Ragdrop",
            path: "Sources/Ragdrop"
        ),
        .testTarget(
            name: "RagdropTests",
            dependencies: ["Ragdrop"],
            path: "Tests/RagdropTests"
        )
    ]
)
