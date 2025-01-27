#import "./head.typ": head
#import "./util.typ": spacer

#let template(
  title: (),
  authors: (),
  abstract: "",
  keywords: (),
  body,
) = {
  if type(title.ja) == "array" {
    set document(
      author: authors.map(a => a.ja),
      title: title.ja.join(""),
      keywords: keywords,
    )
  } else {
    set document(
      author: authors.map(a => a.ja),
      title: title.ja,
    )
  }

  set page(
    columns: 2,
    binding: right,
    margin: (
      inside: 20mm,
      outside: 15mm,
      y: 15mm,
    ),
    paper: "a4",
  )

  set heading(numbering: "1.1.1", supplement: "")
  show heading: it => {
    text(
      font: "Hiragino Kaku Gothic ProN",
      weight: "bold",
      it,
    )
    pad(spacer(), bottom: -1em)
  }

  set par(
    justify: true,
    spacing: 1em,
    first-line-indent: 1em,
  )

  set text(
    font: "Hiragino Mincho ProN",
    size: 8.8pt,
  )

  set list(indent: 1em)
  show list: it => {
    pad(y: 0.1em, it)
  }

  set figure(
    numbering: "1.1.1",
    gap: 10pt,
    supplement: "図",
  )
  show figure: it => {
    it
    spacer()
  }
  show figure.where(kind: table): set figure.caption(position: top)

  place(
    top + center,
    scope: "parent",
    float: true,
    head(
      title: title,
      authors: authors,
      abstract: abstract,
      keywords: keywords,
    ),
  )

  body

  bibliography("../ref.bib", title: "参考文献", style: "ieee")
}

#let i = h(1em)

#let es = h(0.25em, weak: true)

#let tbl(tbl, caption: "") = {
  figure(
    tbl,
    caption: figure.caption(caption, position: top),
    supplement: [表],
    kind: "table",
  )
}

#let img(img, caption: "") = {
  figure(
    img,
    caption: caption,
    supplement: [図],
    kind: "image",
  )
}

