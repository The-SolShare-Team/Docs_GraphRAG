// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SymbolGraphExtractor",
    platforms: [
        .macOS(.v13)
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-docc-symbolkit", branch: "main")
    ],
    targets: [
        .executableTarget(
            name: "SymbolGraphExtractor",
            dependencies: [
                .product(name: "SymbolKit", package: "swift-docc-symbolkit")
            ],
            path: "./"
        )
    ]
)
