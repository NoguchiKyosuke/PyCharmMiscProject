ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.3.

solve: pip install numpy==2.2

---

TypeError: time_stretch() takes 1 positional argument but 2 were given

solve: time_stretch(x, rate) -> time_stretch(x, rate = 1.0)

---

# なぜGlobal Average Pooling(GAP)を使うのか?

参考文献 - [Understanding the Difference Between Flatten() and GlobalAveragePooling2D() in Keras](https://saturncloud.io/blog/understanding-the-difference-between-flatten-and-globalaveragepooling2d-in-keras/)

従来のflatteningでのアプローチではバッチサイズが大きく、パラメータの量がとても大きくなってしまう。\
そこで、GAPを使うことでパラメータの量を減らすことができる。

flatteningはGAPに対して特徴マップ分かけた量のパラメータ量を持つ。(flatteningのパラメータ量 = GAPのパラメータ量 × 特徴マップのサイズ)\

## flatteningとは
多次元テンソルを一次元テンソルに変換する手法。\
バッチサイズを保持したまま次元を畳み込むことを可能としている。\
3×3のインプットに対しては1×9のテンソルに変換する。

## Global Average Poolingとは
テンソルの空間次元を縮小させる手法。\
flatteningと違い、GAPはデータに対して処理を行う。\
データの各入力の特徴マップの平均値を計算することで、より小さなテンソルを出力できる。\
3×3のインプットに対しては1×3のテンソルに変換する。

---

無料の音声データベースのURL
- [無償入手可能な音声コーパス／音声データベースの一覧](https://qiita.com/nakakq/items/74fea8b55d08032d25f9)

---

accuracyを上げる方法: 
- 学習率を調整する。0.0001ぐらいがちょうどよい。
- dropout

---

dropoutとは、正規化のテクニックの一つである。
トレーニングセッションおきに、ランダムでいくつかのニューロンの出力値を0にすることでオーバーフィッテングを防ぐことができる。
またdropoutは現在の一つのニューロンのみに依存しないため、主要な特徴量を抽出するのに強力である。
