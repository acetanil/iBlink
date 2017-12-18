clear

cd ./build

echo '[  INFO] find a proper bounding box...'
./find_bounding_box

echo '[  INFO] success'
echo ''
echo '[  INFO] now begin...'
./main

cd ..

