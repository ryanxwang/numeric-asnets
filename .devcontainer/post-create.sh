# Install ASNets
python3 -m venv venv-asnets && . venv-asnets/bin/activate
pip3 install --upgrade pip
pip3 install wheel cython numpy pkgconfig werkzeug
pip3 install -e asnets

# Install PDDL parser
git clone https://github.com/pucrs-automated-planning/pddl-parser.git
cd pddl-parser && python3 setup.py install && cd ..

# Install PySAT
pip3 install python-sat[pblib,aiger]

# Install Fast Downward
git clone https://github.com/aibasel/downward.git
cd downward && ./build.py && cd ..

export PYTHONPATH=$PYTHONPATH:.