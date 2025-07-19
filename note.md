ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.3.

solve: pip install numpy==2.2

---

TypeError: time_stretch() takes 1 positional argument but 2 were given

solve: time_stretch(x, rate) -> time_stretch(x, rate = 1.0)

---

# なぜGlobal Average Pooling(GAP)を使うのか?

従来のflatteningでのアプローチではバッチサイズが大きく、パラメータの量がとても大きくなってしまう。\
そこで、GAPを使うことでパラメータの量を減らすことができる。

flatteningはGAPに対して特徴マップ分かけた量のパラメータ量を持つ。(flatteningのパラメータ)

## flatteningとは
多次元テンソルを一次元テンソルに変換する手法。\
バッチサイズを保持したまま次元を畳み込むことを可能としている。\
3×3のインプットに対しては1×9のテンソルに変換する。

## Global Average Poolingとは
テンソルの空間次元を縮小させる手法。\
flatteningと違い、GAPはデータに対して処理を行う。\
データの各入力の特徴マップの平均値を計算することで、より小さなテンソルを出力できる。\
3×3のインプットに対しては1×3のテンソルに変換する。
