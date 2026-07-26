#!/usr/bin/env swift

import AppKit
import Foundation
import PDFKit

guard CommandLine.arguments.count == 3 else {
    fputs("usage: render_pdf_pages.swift INPUT.pdf OUTPUT_DIR\n", stderr)
    exit(2)
}

let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

guard let document = PDFDocument(url: inputURL) else {
    fputs("could not open \(inputURL.path)\n", stderr)
    exit(1)
}

let scale: CGFloat = 1.5
for index in 0..<document.pageCount {
    guard let page = document.page(at: index) else { continue }
    let bounds = page.bounds(for: .mediaBox)
    let width = Int(bounds.width * scale)
    let height = Int(bounds.height * scale)
    guard let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: width,
        pixelsHigh: height,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else { continue }

    NSGraphicsContext.saveGraphicsState()
    let context = NSGraphicsContext(bitmapImageRep: bitmap)
    NSGraphicsContext.current = context
    context?.cgContext.setFillColor(NSColor.black.cgColor)
    context?.cgContext.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context?.cgContext.scaleBy(x: scale, y: scale)
    page.draw(with: .mediaBox, to: context!.cgContext)
    context?.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()

    if let data = bitmap.representation(using: .png, properties: [:]) {
        let name = String(format: "slide-%02d.png", index + 1)
        try data.write(to: outputURL.appendingPathComponent(name))
    }
}

print("rendered \(document.pageCount) pages")
