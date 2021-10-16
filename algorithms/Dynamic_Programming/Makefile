some_file: other_file
	echo "This line will run second, because it depends on other file"
	touch some_file
other_file:
	echo "this will run first"
	touch other_file
