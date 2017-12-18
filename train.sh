clear

echo '[  INFO] going through the dataset...'
python t_list_xml.py
echo '[  INFO] success'
echo ''
echo '[  INFO] now begin...'

cd build
make -j

./train

./predict_list

cd ..

