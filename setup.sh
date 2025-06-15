if [ ! -e self-attentive-parser ]; then
  git clone https://github.com/nikitakit/self-attentive-parser &> /dev/null
fi
rm -rf train dev test EVALB/
cp self-attentive-parser/data/02-21.10way.clean ./English.train
cp self-attentive-parser/data/22.auto.clean ./English.dev
cp self-attentive-parser/data/23.auto.clean ./English.test
# The evalb program needs to be compiled
cp -R self-attentive-parser/EVALB EVALB
cd EVALB && make &> /dev/null
# To test that everything works as intended, we check that the F1 score when
# comparing the dev set with itself is 100.
./evalb -p nk.prm ../English.dev ../English.dev | grep FMeasure | head -n 1