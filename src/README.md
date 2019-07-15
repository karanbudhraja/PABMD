variational.py
==============

-> modular variational framework for reverse mapping + forward mapping
-> contains forward mapping (many to one) code
-> contains reverse mapping (one to many) code
-> current architecture seems to work better with the civil violence domain

python3 variational.py <abm> <alp dimension> <slp dimension>
e.g.
python3 variational.py civil_violence 3 3

dataset locations
=================
../../data/domaindata/cross_validation/

terminology
===========
-> alps: x dimension (input parameters to an agent-based model)
-> slps: y dimension (quantified output behavior of agent-based model)
-> forward mapping network: x->y
-> reverse mapping network: y->x + x->y (x-> is used to determine loss, but x is what needs to be given to the user)
