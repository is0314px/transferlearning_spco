pythonが使える環境が必要です。

＜使い方＞
１．input_location_names_python2.pyを実行(python2の場合) Python 2.7.17で動作確認済み
　　input_location_names_python3.pyを実行(python3の場合) Python 3.6.9で動作確認済み
２．ロボットが取得した画像と、そのときの位置（二次元座標と向き）が左側に出力されるので、端末に場所の名前を発話文で入力してENTER（ロボットが目の前にいると想定してください）
	注意事項！！
        ・日本語は使用不可
	・小文字のアルファベットのみ（固有名詞なら大文字でOK）
	・ピリオドとカンマは使用不可
	・固有名詞がついた部屋でもOK
	（入力例）
	・this place is in front of TV
	・you are in Taro's-room

＜必要なライブラリ＞
以下のライブラリが自分のパソコンに入ってなかったらインストールお願いします。
・numpy
・matplotlib
・PIL

-----------------------------------------------------------------------
You need to have the environment that install python

<How to use>
1. Excute bellow file.
   input_location_names_python2.py (if you use python2) Confirmed to work with python 2.7.17
   input_location_names_python3.py (if you use python3) Confirmed to work with python 3.6.9

2. After you excute the file, you find the image and position (2D coordinate and directions) on the left, please input the location name as utterance and press <ENTER> on your terminal. (there is a robot in front of you)
	Note!!
	* You can use only lowercase alphabet (However, you can only use capital letters for proper nouns)
	* Don't use period and comma
	* You can use rooms name included propose nouns with capital letters
	(Input Examples)
	* this place is in front of TV
	* you are in Taro's-room

<Mandatory library>
Please install below libraries if you have not.
* numpy
* matplotlib
* PIL







