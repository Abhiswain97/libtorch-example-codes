if [ -d "./build" ]; then
	echo "Build folder exists, rebuilding and executing"
	cd build
	cmake -DCMAKE_PREFIX_PATH=$1 ..
	make
	./Mnist 
else
	echo "Creating build folder and building the project"
	mkdir build
	cd build
	cmake -DCMAKE_PREFIX_PATH=$1 ..
	make 
	./Mnist
fi
