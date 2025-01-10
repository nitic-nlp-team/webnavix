#import "./template/template.typ": i, es, img, tbl, template

#show: template.with(
  title: (
    ja: (
      "WebNavix：Domain-wise Mixture-of-Expertsによる",
      "継続汎用Web Navigationエージェント",
    ),
    en: "WebNavix: Continual Generalist Web Navigation Agent with Domain-wise Mixture-of-Experts",
  ),
  authors: (
    (ja: "塩畑 晴人", en: "Haruto Shiohata", group: 1),
    (ja: "柴沼 厳", en: "Itsuki Shibanuma", group: 1),
    (ja: "池田 耕", en: "Koh Ikeda", group: 1),
  ),
  affiliations: (
    (ja: "茨城工業高等専門学校", en: "National Institute of Technology, Ibaraki College", group: 1),
  ),
)

= はじめに

#(i)Web Navigationエージェントは、ウェブページの状態とユーザーの指示を入力とし、意思決定エンジンを経て、ウェブページへの操作を出力するシステムである。@deng2023mind2web 最近のWeb Navigationに関する研究は、意思決定エンジンとしての大規模言語モデル（LLM）のIn-context Learningの工夫や、フレームワーク全体の改善にフォーカスされることが多い。一方、十分に探求されていない課題として、エージェントの未知の環境への汎化と、継続学習の効率化が挙げられる。既存のWeb Navigationのためのデータセットは複数のドメインを含んでいるが、これに対して密なLLMがすべてのドメインで高い汎化性能を獲得するのは困難である。また、実世界では、機能の刷新や全く新しいドメインの登場などにより、理想的なデータセットは短期間で更新されるため、エージェントはその変化を継続的に学習する必要がある。しかし、巨大なLLMのフルパラメータFine-tuningには大規模な計算環境が必要であり、一度の継続学習でも莫大なコストが発生する。本論文では、これらの課題を解決するための手法として、データセットをドメインごとに分割し、それぞれのドメインに対して個別にFine-tuningされたLLMをドメイン専門家として継続的に統合する*WebNavix*を提案する。

#let webnavix-footnotes = (
  appropriate-input-representation: "ポジティブプロンプトと呼称され、そのELMが対応するべきプロンプトの具体的な例が与えられる。",
)

= WebNavix

#(i)WebNavixは、Mixture-of-Experts（MoE）@jiang2024mixtral に基づくドメイン別モデルマージ手法であるBranch-Train-MiX（BTX）@sukhbaatar2024branchtrainmixmixingexpertllms をWeb Navigationのために拡張する。WebNavixは、Task Adapt、Branch、Domain Adapt、MiXの4つのステージから構成される。

== Task Adapt

#(i)Task Adaptでは、シードLLM#(es)$cal(M)$#(es)をWeb Navigationに適応させる。Web Navigationのためのデータセット#(es)$cal(D)$#(es)が与えられたとき、Task Adaptでは一度#(es)$cal(D)$#(es)全体で$cal(M)$#(es)をフルパラメータFine-tuningする。これにより、$cal(M)$#(es)内のFeed Forward Network（FFN）、Self-attention、その他すべてのサブレイヤーをWeb Navigationに適応させる。ただし、後のステージでの学習の安定性を考慮して、$cal(M)$#(es)は過学習を避けるために少量のエポック数で学習される。

== Branch

#(i)Branchでは、$cal(D) = {D_i}_(i=1)^N$により#(es)$cal(D)$#(es)が含むドメイン数#(es)$N$#(es)で#(es)$cal(D)$#(es)を分割し、$cal(M)$#(es)によって初期化された#(es)$N $#(es)個のExpert Language Model（ELM）$cal(M)_i$#(es)を生成する。この時点で、$cal(M)$#(es)と各#(es)$cal(M)_i$#(es)は等価である。

== Domain Adapt

#(i)Domain Adaptでは、$D_i$#(es)で#(es)$cal(M)_i$#(es)をFine-tuningし、各#(es)$cal(M)_i$#(es)にドメイン内での汎化性能を獲得させる。このとき、学習するパラメータはFFNのパラメータのみとし、その他のサブレイヤーは凍結して学習される。

== MiX

#(i)MiXでは、$cal(M)$#(es)と各#(es)$cal(M)_i$#(es)をMoEに基づいてマージし、WebNavixの最終モデルを構築する。$"FFN"_i^l$#(es)が#(es)$cal(M)_i$#(es)の#(es)$l$#(es)層目のFFNであるとき、入力表現#(es)$x$#(es)に対する#(es)$l$#(es)層目のMoEサブレイヤー#(es)$"FFN"_"MoE"^l (x)$#(es)は次のように計算される。

$ "FFN"_"MoE"^l (x) = sum_(i=1)^(N) g (W^l x) "FFN"_i^l (x) $

ここで、$W^l$#(es)は線形変換である。また、$g$#(es)は通常スパースな出力を持つルーティング関数であり、実験では#(es)$g (W^l x) := "SoftMax"("Top-k"(W^l x))$#(es)$(upright(k)=2)$#(es)で定義する。Self-attentionやその他のサブレイヤーについては、Domain Adapt時の凍結により$cal(M)$#(es)と各#(es)$cal(M)_i$#(es)で等価であるため、$cal(M)$#(es)のパラメータをそのまま使用する。このマージにより、新しいMoEモデル#(es)$cal(M)_"MoE"$#(es)を得る。ただし、$cal(M)_"MoE"$#(es)は未学習のパラメータ#(es)$W^l$#(es)を持つため、通常#(es)$cal(D)$#(es)によって再度Fine-tuningする必要がある。しかし、一般にMoEモデルのFine-tuningには大規模な計算環境を要するため、効率的な継続学習の観点からこの方法は現実的ではない。そのため、WebNavixでは非学習手法により#(es)$W^l$#(es)を決定する。

$"FFN"_i^l$#(es)に対する適切な入力表現#(es)$x_i$#(es)#footnote(webnavix-footnotes.appropriate-input-representation)#(es)が与えられたとき、$cal(M)$#(es)の#(es)$L$#(es)層目までの各隠れ状態#(es)$h_i^l$#(es)は次のように計算される。

$ [h_i^l]_(l=1)^L = cal(M) (x_i) $

このとき、各層における各系列の隠れ状態は#(es)$h_i^l$#(es)として平均される。これを用いて、$W_l$#(es)は次のように決定される。

$ W^l = [h_i^l]_(i=1)^N $

これにより、MiX全体が非学習手法により実現され、ごく少ないコストでの継続学習が可能となる。以上の手順により、WebNavixは、データセット内のすべてのドメインでの汎化性能の獲得と継続学習の効率化を実現する。

= 実験

#(i)従来手法に対するWebNavixの有効性を定量的に示すために、Web Navigationのための広範なベンチマークであるWebLINX @lù2024weblinx によって評価実験を行う。

#let experiment-setup-footnotes = (
  webnavix: link("https://huggingface.co/collections/nitic-nlp-team/webnavix-6733e035c7b41d86090b2bf3"),
  i: "AI Tools、Booking、Composing、Information Lookup、Shopping、Social Interaction、Summarizing、Task Management",
)

== 実験設定

#(i)本実験における評価対象として、以下のパラメータをもとにWebNavixモデルを構築する。#h(-0.2em, weak: true)#footnote(experiment-setup-footnotes.webnavix)

- $cal(D)$：WebLINX
- $cal(M)$：Sheared-LLaMA-2.7B @xia2024shearedllamaacceleratinglanguage（従来SoTA）
- $i$：WebLINXの各カテゴリ#h(0.1em, weak: true)#footnote(experiment-setup-footnotes.i)（$N=8$）

これに対し、WebLINXが提供するメトリクスに従い、$#smallcaps(text("Test", font: "New Computer Modern"))_text("IID")$#(es)のインテント・マッチ（IM）、エレメント・グループ（IoU）、テキスト・グループ（F1）、総合スコア（ターンレベルのスコアのマイクロ平均）の結果を報告する。比較対象として、いずれかの項目でSoTAを持つモデルのスコアをWebLINX論文から引用する。なお、予算の制限により、WebNavixモデルは4bit量子化により推論した結果を報告する。

== 結果

#(i)主な結果を@main-results\に示す。本実験におけるWebNavixは量子化された推論であるにもかかわらず、総合スコアのSoTAを7.48%更新した。併せて、IoUは31.26%、F1は6.59%向上した一方、IMでは2.16%の低下が見られた。

== ルーティング解析

#(i)WebNavixの性能をより深く調査するために、実験で構築したWebNavixモデルのルーティング解析を行う。ルーティング解析は、WebLINXの各カテゴリに属する入力テキストに対するWebNavixモデルの各層のルーティング結果を可視化することで行う。
理想的には、各層で入力テキストのドメインと割当てられたELMのドメインが一致することが期待されるが、@routing-analysis\に示すように、WebNavixモデルのゲーティングネットワークは入力テキストのドメインに一切影響されず、層に影響されてルーティングを決定していることが確認された。この原因として、Task Adapt時のepoch数が多く、その時点でデータセットを十分に学習してしまい、Domain Adapt時の学習で各ELMの能力に十分な差が生じなかったことが考えられる。

= 課題と展望

#(i)Web Navigationのためのモデルは、システムの機密性や可用性の観点から、ユーザーのマシン上で動作可能であるように十分に軽量であることが望まれる。しかし、本実験で構築したWebNavixモデルのサイズは14.6Bであり、これは量子化などによりモデルサイズを圧縮しなければ、一般的なPCやスマートフォン上で動作させることはできない。

また、WebNavixはHTMLからウェブページの情報を抽出するが、この方法では画像コンテンツのような視覚的要素の情報をモデルが得ることはできない。従って、今後の研究では、WebNavixの視覚言語モデルへの応用が望まれる。

= おわりに

#let conclusion-footnotes = (
  project-url: link("https://github.com/nitic-nlp-team/webnavix"),
)

#(i)本論文では、個別に学習されたドメイン専門家としてのLLMを継続的に統合するMoEに基づくドメイン別モデルマージの方法であるWebNavixを、初めてWeb Navigationに導入した。実験では、WebLINXによる評価において、WebNavixが最近の手法と比較して優れた性能を示すことが確認された。この成功の一方で、WebNavixは、学習時に肥大化したモデルサイズの圧縮や、視覚言語モデルへの応用などの、実用化に際して解決しなければならない課題を多数残している。

#place(
  top + center,
  float: true,
  [
    #set text(size: 0.8em)
    #tbl(
      table(
        columns: 6,
        row-gutter: (0.2em, auto),
        stroke: (x: none, y: 0.8pt),
        table.header(
          text()[*Models*],
          text()[*Size*],
          text()[*IM*],
          text()[*IoU*],
          text()[*F1*],
          smallcaps(
            text(
              "Overall",
              font: "New Computer Modern",
              weight: "bold",
            ),
          ),
        ),

        text()[S-LLaMA], text()[2.7B], text()[*87.70*], text()[35.54], text()[37.66], text()[37.43],
        text()[Llama-2], text()[13B], text()[*87.70*], text()[35.92], text()[37.43], text()[37.09],
        text()[WebNavix#h(0.2em)#super[*Q*]],
        text()[14.6B],
        text()[85.81],
        text()[*47.15*],
        text()[*40.14*],
        text()[*40.23*],
      ),
      caption: [WebLINXに基づく評価の主な結果。予算の制限により、WebNavixは4bit量子化により推論した結果を報告する。],
    ) <main-results>
  ],
)

#place(
  top + center,
  dy: -0.8em,
  clearance: 1.25em,
  float: true,
  [
    #set text(size: 0.8em)
    #img(
      grid(
        columns: 2,
        image("./image/routing-analysis-layer-0.svg"), image("./image/routing-analysis-layer-7.svg"),
        image("./image/routing-analysis-layer-15.svg"), image("./image/routing-analysis-layer-31.svg"),
      ),
      caption: [WebNavixモデルのルーティング解析結果。左上が1層目、右上が8層目、左下が16層目、右下が32層目のルーティング結果を示す。],
    ) <routing-analysis>
  ],
)
