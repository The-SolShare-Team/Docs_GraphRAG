import Foundation
import SymbolKit

struct EnrichedSymbol: Encodable {
    let id: String
    let moduleName: String
    
    
    let essential: SymbolExtractors.EssentialMetadata
    let generics: SymbolExtractors.GenericMetadata
    let function: SymbolExtractors.FunctionMetadata
    let variable: SymbolExtractors.VariableMetadata
    let relationships: SymbolExtractors.RelationshipMetadata

    

    enum CodingKeys: String, CodingKey {
        case id, name, kind, type, absolutePath, filePath, accessLevel, moduleName
        case relationships
        case cleanDocString, cleanDeclaration, cleanGenerics
        case functionSignature, parameterNames, returnType
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        func encode(_ value: String?, forKey key: CodingKeys) throws {
            // Check if nil OR empty string
            if let v = value, !v.isEmpty {
                try container.encode(v, forKey: key)
            }
        }

        func encode(_ value: [String]?, forKey key: CodingKeys) throws {
            // Check if nil OR empty array
            if let v = value, !v.isEmpty {
                try container.encode(v, forKey: key)
            }
        }

        // 1. Top Level
        try encode(id, forKey: .id)
        try encode(moduleName, forKey: .moduleName)

        // 2. Essential Metadata
        try encode(essential.name, forKey: .name)
        try encode(essential.kind.lowercased(), forKey: .kind)
        try encode(essential.absolutePath, forKey: .absolutePath)
        try encode(essential.filePath, forKey: .filePath)
        try encode(essential.accessLevel, forKey: .accessLevel)
        try encode(essential.docText, forKey: .cleanDocString)
        try encode(essential.declaration, forKey: .cleanDeclaration)

        // 3. Variable Metadata
        try encode(variable.typeAnnotation, forKey: .type)

        // 4. Generics Metadata
        try encode(generics.whereClause, forKey: .cleanGenerics)

        // 5. Function Metadata
        try encode(function.signature, forKey: .functionSignature)
        try encode(function.returnType, forKey: .returnType)
        try encode(function.parameterNames, forKey: .parameterNames) // Handles empty array automatically via helper

        // 6. Relationships
        try encode(relationships.descriptions, forKey: .relationships)
    }
}

enum SymbolExtractors {

    struct EssentialMetadata {
        let name: String
        let kind: String
        let accessLevel: String
        let absolutePath: String
        let filePath: String?
        let docText: String
        let declaration: String
    }
    
    static func extractEssential(from symbol: SymbolGraph.Symbol, moduleName: String) -> EssentialMetadata {
        let name = symbol.names.title
        let kind = symbol.kind.displayName
        let accessLevel = symbol.accessLevel.rawValue
        let absolutePath = symbol.absolutePath 
        
        var filePath: String?
        if let locMixin = symbol.mixins[SymbolGraph.Symbol.Location.mixinKey] as? SymbolGraph.Symbol.Location,
           let url = URL(string: locMixin.uri) {
            
            let fullPath = url.path
            // 1. Preferred: Anchor to "/Sources/" (Standard Swift Layout)
            // Result: "PackageName/Sources/ModuleName/File.swift"
            if let range = fullPath.range(of: "/Sources/") {
                let preSourcesPath = String(fullPath[..<range.lowerBound])
                let packageName = URL(fileURLWithPath: preSourcesPath).lastPathComponent
                let postSourcesPath = String(fullPath[range.upperBound...])
                filePath = "\(packageName)/Sources/\(postSourcesPath)"
            }
            // 2. Fallback: Anchor to the Module Name if "Sources" is missing
            // Result: "ModuleName/File.swift"
            else if let range = fullPath.range(of: "/\(moduleName)/") {
                filePath = String(fullPath[range.lowerBound...].dropFirst()) // drop leading slash
            }
            // 3. Last Resort: Use the full path (or handle Tests folder similarly)
            else {
                filePath = fullPath
            }
        }

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
            
        return EssentialMetadata(
            name: name,
            kind: kind,
            accessLevel: accessLevel,
            absolutePath: absolutePath,
            filePath: filePath,
            docText: docText,
            declaration: decl
        )
    }

    struct GenericMetadata {
        let rawConstraints: [String]
        let whereClause: String?
    }

    static func extractGenerics(from symbol: SymbolGraph.Symbol) -> GenericMetadata {
        guard let generics = symbol.mixins[SymbolGraph.Symbol.Swift.Generics.mixinKey] as? SymbolGraph.Symbol.Swift.Generics else {
            return GenericMetadata(rawConstraints: [], whereClause: nil)
        }
        
        let constraints = generics.constraints
            .map { "\($0.leftTypeName) \($0.kind.rawValue) \($0.rightTypeName)" }
        
        let whereClause = constraints.isEmpty ? nil : "where " + constraints.joined(separator: ", ")
        
        return GenericMetadata(rawConstraints: constraints, whereClause: whereClause)
    }

    struct FunctionMetadata {
        let signature: String?
        let parameterNames: [String]?
        let returnType: String?
    }

    static func extractFunctionDetails(from symbol: SymbolGraph.Symbol) -> FunctionMetadata {
        guard let signatureMixin = symbol.mixins[SymbolGraph.Symbol.FunctionSignature.mixinKey] as? SymbolGraph.Symbol.FunctionSignature else {
            return FunctionMetadata(signature: nil, parameterNames: nil, returnType: nil)
        }
        
        let params = signatureMixin.parameters.map { param in
            let typeStr = param.declarationFragments.map { $0.spelling }.joined()
            return "\(param.name): \(typeStr)"
        }
        let paramNames = signatureMixin.parameters.map { $0.name }
        let returnStr = signatureMixin.returns.map { $0.spelling }.joined()
        let returnType = returnStr.isEmpty ? "Void" : returnStr
        let signature = "(\(params.joined(separator: ", "))) -> \(returnType)"
        
        return FunctionMetadata(signature: signature, parameterNames: paramNames, returnType: returnType)
    }

    struct VariableMetadata {
        let typeAnnotation: String?
    }

    static func extractVariableType(from declaration: String, kind: String) -> VariableMetadata {
        guard kind.lowercased().contains("property") || kind.lowercased().contains("var") else {
            return VariableMetadata(typeAnnotation: nil)
        }
        guard let colonRange = declaration.range(of: ": ") else {
            return VariableMetadata(typeAnnotation: nil)
        }
        
        let typeStr = String(declaration[declaration.index(after: colonRange.lowerBound)...])
            .trimmingCharacters(in: .whitespaces)
        
        return VariableMetadata(typeAnnotation: typeStr)
    }

    struct RelationshipMetadata {
        let descriptions: [String]
    }
    
    static func buildRelationships(for symbolID: String, in graph: UnifiedSymbolGraph) -> RelationshipMetadata {
        var relationships: Set<String> = []

        for (_, rels) in graph.relationshipsByLanguage {
            for rel in rels {
                guard rel.source == symbolID else { continue }
                let targetName = rel.target
                
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
        return RelationshipMetadata(descriptions: Array(relationships).sorted())
    }
}

@main
struct SymbolGraphExtractor {

    static func main() async {
        let (inputPath, debugOutputPath) = parseArguments()
        let collector = GraphCollector()

        loadAndMergeGraphs(from: inputPath, into: collector)

        var outputData: [EnrichedSymbol] = []
        let (unifiedGraphs, _) = collector.finishLoading()

        for (_, unifiedGraph) in unifiedGraphs {
            let moduleName = unifiedGraph.moduleName
            
            for (id, unifiedSymbol) in unifiedGraph.symbols {
                guard let symbol = unifiedSymbol.effectiveSymbol(prioritizing: "ios"),
                      let primarySelector = getPrimarySelector(from: unifiedSymbol) else { continue }

                // --- Clean Extraction ---
                let essential = SymbolExtractors.extractEssential(from: symbol, moduleName: moduleName)
                let generics = SymbolExtractors.extractGenerics(from: symbol)
                let relations = SymbolExtractors.buildRelationships(for: id, in: unifiedGraph)
                let funcDetails = SymbolExtractors.extractFunctionDetails(from: symbol)
                let varDetails = SymbolExtractors.extractVariableType(from: essential.declaration, kind: essential.kind)

                guard unifiedSymbol.modules[primarySelector] != nil else { continue }

                let enriched = EnrichedSymbol(
                    id: id,
                    moduleName: moduleName,
                    essential: essential,
                    generics: generics,
                    function: funcDetails,
                    variable: varDetails,
                    relationships: relations
                )
                
                outputData.append(enriched)
            }
        }
        
        saveOutput(outputData, to: debugOutputPath)
    }
    static func parseArguments() -> (inputPath: String, outputPath: String) {
        let inputPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "../symbol-graphs"
        let debugOutputPath = "../enriched_symbols.json"
        return (inputPath, debugOutputPath)
    }

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
                if ignoreList.contains(file.lastPathComponent) { continue }
                let fileData = try Data(contentsOf: file)
                let graph = try JSONDecoder().decode(SymbolGraph.self, from: fileData)
                collector.mergeSymbolGraph(graph, at: file)
            }
        } catch {
            printError("Failed to load symbol graphs: \(error)")
            exit(1)
        }
    }
    
    static func getPrimarySelector(from unifiedSymbol: UnifiedSymbolGraph.Symbol) -> UnifiedSymbolGraph.Selector? {
        return unifiedSymbol.mainGraphSelectors.first {
            $0.platform?.lowercased() == "ios"
        } ?? unifiedSymbol.mainGraphSelectors.first
    }

    static func saveOutput(_ data: [EnrichedSymbol], to debugOutputPath: String) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        
        do {
            let jsonData = try encoder.encode(data)
            
            let debugURL = URL(fileURLWithPath: debugOutputPath)
            try jsonData.write(to: debugURL)
            printError("Debug JSON saved to: \(debugOutputPath)")
            
            if let jsonString = String(data: jsonData, encoding: .utf8) { print(jsonString) }
        } catch {
            printError("Failed to encode/save final JSON: \(error)")
            exit(1)
        }
    }
        
    static func printError(_ msg: String) {
        if let data = "Log: \(msg)\n".data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}

extension UnifiedSymbolGraph.Symbol {
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