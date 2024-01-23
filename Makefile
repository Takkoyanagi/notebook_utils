target: package

package: package
	python3 -m pip install --upgrade build
	python3 -m build
