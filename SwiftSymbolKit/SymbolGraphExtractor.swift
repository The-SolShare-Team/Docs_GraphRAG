
import Foundation
import SymbolKit

@main
struct SymbolGraphExtractor {

    struct EnrichedSymbol: Encodable {
        let id: String
        let name: String
        let kind: String // e.g., "class", "func", "var"
        let type: String? // The type of a variable/property
        let filePath: String?

        let moduleName: String
        // let module: SymbolGraph.Module
        
        let relationships: [String]
        let cleanDocString: String
        let cleanDeclaration: String
        let cleanGenerics: String?

        let functionSignature: String?
        let parameterNames: [String]?
        let returnType: String?
    }

    // MARK: - Main Entry Point

    static func main() async {
        let (inputPath, debugOutputPath) = parseArguments()
        let collector = GraphCollector()

        loadAndMergeGraphs(from: inputPath, into: collector)

        var outputData: [EnrichedSymbol] = []
        let (unifiedGraphs, _) = collector.finishLoading()

        for (_, unifiedGraph) in unifiedGraphs {
            for (id, unifiedSymbol) in unifiedGraph.symbols {
                guard let symbol = unifiedSymbol.effectiveSymbol(prioritizing: "ios"),
                      let primarySelector = getPrimarySelector(from: unifiedSymbol) else { continue }

                // --- Use helper functions to extract data cleanly ---
                let essentialData = extractEssentialData(from: symbol)
                let generics = extractGenericConstraints(from: symbol)
                let relationships = buildRelationships(for: id, in: unifiedGraph, selector: primarySelector)
                let functionDetails = extractFunctionDetails(from: symbol, kind: essentialData.kind)
                
                // Extract type only for variables/properties
                let type: String? = (essentialData.kind.contains("Property")) ? extractVariableType(from: essentialData.decl) : nil

                guard let module = unifiedSymbol.modules[primarySelector] else { continue }
                let moduleName = unifiedGraph.moduleName

                let enriched = EnrichedSymbol(
                    id: id,
                    name: essentialData.name,
                    kind: essentialData.kind,
                    type: type,
                    filePath: essentialData.filePath,
                    moduleName: moduleName,
                    // module: module,
                    relationships: relationships,
                    cleanDocString: essentialData.docText,
                    cleanDeclaration: essentialData.decl,
                    cleanGenerics: generics,
                    functionSignature: functionDetails.functionSignature,
                    parameterNames: functionDetails.parameterNames,
                    returnType: functionDetails.returnType
                )
                
                outputData.append(enriched)
            }
        }
        
        saveOutput(outputData, to: debugOutputPath)
    }

    // MARK: - Helper Functions

    /// Parses command-line arguments for input and output paths.
    static func parseArguments() -> (inputPath: String, outputPath: String) {
        let inputPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "../symbol-graphs"
        let debugOutputPath = "../enriched_symbols.json"
        return (inputPath, debugOutputPath)
    }

    /// Discovers, decodes, and merges all symbol graph files from a given path into the collector.
    static func loadAndMergeGraphs(from inputPath: String, into collector: GraphCollector) {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false

        guard fileManager.fileExists(atPath: inputPath, isDirectory: &isDirectory) else {
            printError("Input path not found: \(inputPath)")
            exit(1)
        }
        let ignoreList: Set<String> = [
            "_RopeModule.symbols.json",
            "_RopeModule@Swift.symbols.json",
            "BitCollections.symbols.json",
            "BitCollections@Swift.symbols.json",
            "ByteBuffer.symbols.json",
            "Collections.symbols.json",
            "Collections@Swift.symbols.json",
            "DequeModule.symbols.json",
            "HashTreeCollections.symbols.json",
            "HeapModule.symbols.json",
            "InternalCollectionsUtilities.symbols.json",
            "InternalCollectionsUtilities@Swift.symbols.json",
            "OrderedCollections.symbols.json",
        ]

        do {
            let files: [URL]
            if isDirectory.boolValue {
                files = try fileManager.contentsOfDirectory(atPath: inputPath)
                    .filter { $0.hasSuffix(".symbols.json") }
                    .map { URL(fileURLWithPath: inputPath).appendingPathComponent($0) }
            } else {
                files = [URL(fileURLWithPath: inputPath)]
            }
            
            for file in files {
                if ignoreList.contains(file.lastPathComponent) {
                    printError("Skipping ignored symbol graph: \(file.lastPathComponent)")
                    continue
                }
                let fileData = try Data(contentsOf: file)
                let graph = try JSONDecoder().decode(SymbolGraph.self, from: fileData)
                collector.mergeSymbolGraph(graph, at: file)
            }
        } catch {
            printError("Failed to load symbol graphs: \(error)")
            exit(1)
        }
    }

    /// Extracts common, high-level data from a symbol.
    static func extractEssentialData(from symbol: SymbolGraph.Symbol) -> (name: String, kind: String, filePath: String?, docText: String, decl: String) {
        let name = symbol.names.title
        let kind = symbol.kind.displayName
        let filePath = symbol.absolutePath
        
        let docText = symbol.docComment?.lines
            .map { $0.text }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        let decl = (symbol.mixins[SymbolGraph.Symbol.DeclarationFragments.mixinKey] as? SymbolGraph.Symbol.DeclarationFragments)?
            .declarationFragments
            .map { $0.spelling }
            .joined()
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            ?? symbol.names.title
            
        return (name, kind, filePath, docText, decl)
    }
    
    /// Extracts the generic constraints from a symbol, if they exist.
    static func extractGenericConstraints(from symbol: SymbolGraph.Symbol) -> String? {
        guard let generics = symbol.mixins[SymbolGraph.Symbol.Swift.Generics.mixinKey] as? SymbolGraph.Symbol.Swift.Generics else {
            return nil
        }
        let constraints = generics.constraints
            .map { "\($0.leftTypeName) \($0.kind.rawValue) \($0.rightTypeName)" }
        if !constraints.isEmpty {
            return "where " + constraints.joined(separator: ", ")
        }
        return nil
    }

    /// Extracts detailed information from a function's signature.
    static func extractFunctionDetails(from symbol: SymbolGraph.Symbol, kind: String) -> (functionSignature: String?, parameterNames: [String]?, returnType: String?) {
        guard let signatureMixin = symbol.mixins[SymbolGraph.Symbol.FunctionSignature.mixinKey] as? SymbolGraph.Symbol.FunctionSignature else {
            return (nil, nil, nil)
        }
        
        let params = signatureMixin.parameters.map { param in
            let typeStr = param.declarationFragments.map { $0.spelling }.joined()
            return "\(param.name): \(typeStr)"
        }
        let parameterNames = signatureMixin.parameters.map { $0.name }
        let returnStr = signatureMixin.returns.map { $0.spelling }.joined()
        let returnType = returnStr.isEmpty ? "Void" : returnStr
        let functionSignature = "(\(params.joined(separator: ", "))) -> \(returnType)"
        
        return (functionSignature, parameterNames, returnType)
    }

    /// Extracts the type of a variable/property from its declaration string.
    static func extractVariableType(from declaration: String) -> String? {
        guard let colonRange = declaration.range(of: ": ") else { return nil }
        return String(declaration[declaration.index(after: colonRange.lowerBound)...]).trimmingCharacters(in: .whitespaces)
    }
    
    /// Finds the best primary selector for a unified symbol, favoring iOS.
    static func getPrimarySelector(from unifiedSymbol: UnifiedSymbolGraph.Symbol) -> UnifiedSymbolGraph.Selector? {
        return unifiedSymbol.mainGraphSelectors.first {
            $0.platform?.lowercased() == "ios"
        } ?? unifiedSymbol.mainGraphSelectors.first
    }

    /// Encodes the final data and writes it to both a debug file and stdout.
    static func saveOutput(_ data: [EnrichedSymbol], to debugOutputPath: String) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        
        do {
            let jsonData = try encoder.encode(data)
            
            let debugURL = URL(fileURLWithPath: debugOutputPath)
            try jsonData.write(to: debugURL)
            printError("Debug JSON saved to: \(debugOutputPath)")
            
            if let jsonString = String(data: jsonData, encoding: .utf8) {
                print(jsonString)
            }
        } catch {
            printError("Failed to encode/save final JSON: \(error)")
            exit(1)
        }
    }
    
    // MARK: - Existing Helpers (Unchanged)

    static func buildRelationships(for symbolID: String, in graph: UnifiedSymbolGraph, selector: UnifiedSymbolGraph.Selector) -> [String] {
        // ... (this function is already well-structured and remains the same)
        var relationships: Set<String> = []

        func getFullyQualifiedName(for id: String) -> String {
            guard let unifiedSymbol = graph.symbols[id],
                let pathComponents = unifiedSymbol.pathComponents[selector] else {
                return id
            }
            return pathComponents.joined(separator: ".")
        }

        for (_, rels) in graph.relationshipsByLanguage {
            for rel in rels {
                guard rel.source == symbolID else { continue }
                let targetName = getFullyQualifiedName(for: rel.target)
                
                let description: String
                switch rel.kind {
                case .memberOf: description = "Is a member of: \(targetName)"
                case .optionalMemberOf: description = "Is an optional member of: \(targetName)"
                case .inheritsFrom: description = "Inherits from: \(targetName)"
                case .conformsTo: description = "Conforms to: \(targetName)"
                case .overrides: description = "Overrides: \(targetName)"
                case .requirementOf: description = "Is a requirement of: \(targetName)"
                case .optionalRequirementOf: description = "Is an optional requirement of: \(targetName)"
                case .defaultImplementationOf: description = "Is a default implementation of: \(targetName)"
                case .extensionTo: description = "Extends: \(targetName)"
                case .references: description = "References: \(targetName)"
                case .overloadOf: description = "Is an overload of: \(targetName)"
                default: continue
                }
                relationships.insert(description)
            }
        }
        return Array(relationships).sorted()
    }
        
    static func printError(_ msg: String) {
        if let data = "Log: \(msg)\n".data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}

// MARK: - SymbolKit Extension (Unchanged)

extension UnifiedSymbolGraph.Symbol {
    /// Reconstructs a standard `SymbolGraph.Symbol` view from the unified data
    /// by picking the best matching selector (e.g. favoring "ios").
    func effectiveSymbol(prioritizing platform: String = "ios") -> SymbolGraph.Symbol? {
        let selector = self.mainGraphSelectors.first {
            $0.platform?.lowercased() == platform.lowercased()
        } ?? self.mainGraphSelectors.first
        
        guard let bestSelector = selector else { return nil }
        
        guard let kind = self.kind[bestSelector],
              let names = self.names[bestSelector],
              let pathComponents = self.pathComponents[bestSelector],
              let accessLevel = self.accessLevel[bestSelector] else {
            return nil
        }
        
        let identifier = SymbolGraph.Symbol.Identifier(
            precise: self.uniqueIdentifier,
            interfaceLanguage: bestSelector.interfaceLanguage
        )
        
        return SymbolGraph.Symbol(
            identifier: identifier,
            names: names,
            pathComponents: pathComponents,
            docComment: self.docComment[bestSelector],
            accessLevel: accessLevel,
            kind: kind,
            mixins: self.mixins[bestSelector] ?? [:]
        )
    }
}