# Just file: https://github.com/casey/just

test:
    pytest

release version: test
    git tag {{version}}
    git checkout {{version}}
    git push --tags -f
    python3 setup.py sdist
    twine upload dist/pyronn-torch-{{version}}.tar.gz
    git checkout master
    
wheel_release version: test
    git tag {{version}}
    git checkout {{version}}
    git push --tags -f
    python3 setup.py bdist_wheel
    twine upload dist/pyronn-torch-{{version}}-cp37-cp37m-linux_x86_64.whl
    git checkout master
