import re
import os
from typing import Iterable, Optional
import weakref


__all__ = [
    'HList', 'parse_sexprs', 'hlist_to_sexprs', 'relax_numeric_pddls',
]


def _ppddl_tokenize(ppddl_txt):
    """Break PPDDL into tokens (brackets, non-bracket chunks)"""
    # strip comments
    lines = ppddl_txt.splitlines()
    mod_lines = []
    for line in lines:
        try:
            semi_idx = line.index(';')
        except ValueError:
            pass
        else:
            line = line[:semi_idx]
        mod_lines.append(line)
    ppddl_txt = '\n'.join(mod_lines)

    # convert to lower case
    ppddl_txt = ppddl_txt.lower()

    matches = re.findall(r'\(|\)|[^\s\(\)]+', ppddl_txt)

    return matches


def _hlist_to_tokens(hlist):
    """Convert a HList back into tokens (either single open/close parens or
    non-paren chunks)"""
    tokens = ['(']
    for item in hlist:
        if isinstance(item, HList):
            tokens.extend(_hlist_to_tokens(item))
        else:
            assert isinstance(item, str), "Can't handle item '%r'" % (item, )
            tokens.append(item)
    tokens.append(')')
    return tokens


class HList(list):
    """Class for hierarchical list. Helpful because you can get at parent from
    current node (or see that current node is root if no parent)."""

    def __init__(self, parent):
        super()
        self.is_root = parent is None
        self._parent_ref = weakref.ref(parent) if not self.is_root else None

    @property
    def parent(self):
        if self.is_root:
            return None
        return self._parent_ref()


def parse_sexprs(ppddl_txt):
    """Hacky parse of sexprs from ppddl."""
    tokens = _ppddl_tokenize(ppddl_txt)
    parse_root = parse_ptr = HList(None)
    # we parse begin -> end
    # just reverse so that pop() is efficient
    tokens_reverse = tokens[::-1]
    while tokens_reverse:
        token = tokens_reverse.pop()
        if token == '(':
            # push
            new_ptr = HList(parse_ptr)
            parse_ptr.append(new_ptr)
            parse_ptr = new_ptr
        elif token == ')':
            # pop
            parse_ptr = parse_ptr.parent
        else:
            # add
            parse_ptr.append(token)
    return parse_root


def hlist_to_sexprs(hlist, indent=None):
    """Convert a HList back to (some semblance of) PDDL."""
    assert isinstance(hlist, HList), \
        "are you sure you want to pass in type %s?" % (type(hlist),)
    tok_stream = _hlist_to_tokens(hlist)

    cur_indent = 0

    out_parts = []
    # was the last token an open paren?
    last_open = True
    for token in tok_stream:
        is_open = token == '('
        is_close = token == ')'
        is_paren = is_open or is_close
        if (not is_paren and not last_open) or (is_open and not last_open):
            # we insert space between token seqs of the form [<non-paren>,
            # <non-paren>] and token seqs of the form [")", "("]
            out_parts.append(' ')
        out_parts.append(token)
        # for next iter
        last_open = is_open

        if is_close and indent is not None:
            cur_indent -= indent
            out_parts.append('\n')
            out_parts.append(' ' * cur_indent)
        if is_open and indent is not None:
            cur_indent += indent

    return ''.join(out_parts)


def get_domain_file(pddl_files: Iterable[str]) \
        -> Optional[str]:
    """Get the first domain file in the given list of PDDL files.

    Args:
        pddl_files (Iterable[str]): List of PDDL files to search.

    Returns:
        Optional[str]: The file containing the first domain in the
        list of PDDL files, or None if no domain is found.
    """
    for pddl_file in pddl_files:
        with open(pddl_file, 'r') as fp:
            pddl_txt = fp.read()
            sexprs = parse_sexprs(pddl_txt)
            for declr in sexprs:
                assert len(declr) >= 2 and declr[0] == "define", \
                    "expected (define …), got AST %s" % (declr, )
                declr_type, declr_name = declr[1]
                if declr_type == "domain":
                    return pddl_file

    return None


def get_problem_file(pddl_files: Iterable[str], problem_name: str) \
        -> Optional[str]:
    """Get the file containing the problem with the given name.

    Args:
        pddl_files (Iterable[str]): List of PDDL files to search.
        problem_name (str): Name of the problem to search for.

    Returns:
        Optional[str]: The file containing the problem with the given
        problem name, or None if no such problem exists.
    """
    for pddl_file in pddl_files:
        with open(pddl_file, 'r') as fp:
            pddl_txt = fp.read()
            sexprs = parse_sexprs(pddl_txt)
            for declr in sexprs:
                assert len(declr) >= 2 and declr[0] == "define", \
                    "expected (define …), got AST %s" % (declr, )
                declr_type, declr_name = declr[1]
                if declr_type == "problem" and declr_name == problem_name:
                    return pddl_file

    return None


def extract_all_domains_problems(pddl_files):
    # make an index of domains & problems by parsing each file in turn
    domains = {}
    problems = {}
    for pddl_file in pddl_files:
        with open(pddl_file, 'r') as fp:
            pddl_txt = fp.read()
        # Each parsed file is list of domains/problems. Domain has the form:
        #
        #  ["define", ["domain", <dname>], …]
        #
        # Problem has the form:
        #
        #  ["define", ["problem", <pname>], …, [":domain", <dname>], …]
        sexprs = parse_sexprs(pddl_txt)
        for declr in sexprs:
            assert len(declr) >= 2 and declr[0] == "define", \
                "expected (define …), got AST %s" % (declr, )
            declr_type, declr_name = declr[1]
            if declr_type == "problem":
                problems[declr_name] = declr
            elif declr_type == "domain":
                domains[declr_name] = declr
            else:
                raise ValueError("Unknown type '%s'" % (declr_type,))
    return domains, problems


def extract_domain_problem(pddl_files, problem_name=None):
    """Extract HLists representing PDDL for domain & problem from a collection
    of PDDL files & a problem name."""
    domains, problems = extract_all_domains_problems(pddl_files)

    # retrieve hlist for problem & figure out corresponding domain
    if problem_name is None:
        problem_names = list(problems.keys())
        if len(problem_names) != 1:
            raise ValueError(
                "Was not given a problem name, and the given PDDL files "
                "contain either 0 or > 1 names (full list: %s)" %
                (problem_names,))
        problem_name, = problem_names
    problem_hlist = problems[problem_name]
    for subpart in problem_hlist:
        if len(subpart) == 2 and subpart[0] == ':domain':
            domain_name = subpart[1]
            break
    else:
        raise ValueError("Could not find domain for '%s'" % (problem_name, ))
    domain_hlist = domains[domain_name]

    return domain_hlist, domain_name, problem_hlist, problem_name


def extract_domain_name(pddl_path):
    """Extract a domain name from a single PDDL domain file."""
    assert isinstance(pddl_path, str), \
        "this only takes a single (string) filename"
    domains, _ = extract_all_domains_problems([pddl_path])
    assert len(domains) == 1, \
        "PDDL file at '%s' contains %d domains (not 1); they are %s" \
        % (pddl_path, len(domains), sorted(domains))
    domain_name, = domains.keys()
    return domain_name


def relax_numeric_domain(domain_hlist: HList) -> HList:
    """Generate the relaxation of a numeric planning domain with all numeric
    components removed.

    Args:
        domain_hlist (HList): the HList of the domain to relax
    
    Returns:
        HList: the relaxed domain.
    """
    new_hlist = HList(parent=None)
    for subsec in domain_hlist:
        if len(subsec) >= 0 and subsec[0] == ':functions':
            continue

        if len(subsec) >= 0 and subsec[0] == ':requirements':
            req_hlist = HList(parent=new_hlist)

            for req in subsec:
                if req in [':fluents']:
                    continue
                req_hlist.append(req)
            
            new_hlist.append(req_hlist)
            continue

        if len(subsec) == 0 or subsec[0] != ':action':
            new_hlist.append(subsec)
            continue
    
        # subsec is an action
        action_hlist = HList(parent=new_hlist)
        action_hlist.extend(subsec[:5])

        preconditions = subsec[5]
        pre_hlist = HList(parent=action_hlist)
        for pre in preconditions:
            if type(pre) == str or pre[0] not in ['=', '<=', '>=', '<', '>']:
                pre_hlist.append(pre)
        action_hlist.append(pre_hlist)

        action_hlist.append(subsec[6])
        effects = subsec[7]
        eff_hlist = HList(parent=action_hlist)
        for eff in effects:
            if type(eff) == str or eff[0] not in ['increase', 'decrease', 'assign']:
                eff_hlist.append(eff)
        action_hlist.append(eff_hlist)
            
        
        new_hlist.append(action_hlist)

    return new_hlist
    

def relax_numeric_problem(problem_hlist: HList) -> HList:
    """Generate the relaxation of a numeric planning problem with all numeric
    components removed.

    Args:
        problem_hlist (HList): the HList of the problem to relax
    
    Returns:
        HList: the relaxed problem.
    """
    new_hlist = HList(parent=None)
    for subsec in problem_hlist:
        if len(subsec) >= 1 and subsec[0] == ':init':
            init_hlist = HList(parent=new_hlist)
            init_hlist.append(":init")
            for atom in subsec[1:]:
                if type(atom) == str or atom[0] not in ['=']:
                    init_hlist.append(atom)
            new_hlist.append(init_hlist)
            continue

        if len(subsec) >= 1 and subsec[0] == ':goal':
            goal_hlist = HList(parent=new_hlist)
            goal_hlist.append(":goal")
            for atom in subsec[1:]:
                if type(atom) == str or atom[0] not in ['=', '<=', '>=', '<', '>']:
                    goal_hlist.append(atom)
            new_hlist.append(goal_hlist)
            continue
    
        if len(subsec) >= 1 and subsec[0] == ':metric':
            continue

        new_hlist.append(subsec)

    return new_hlist


def relax_numeric_pddls(pddl_paths: Iterable[str]) -> Iterable[str]:
    """Generate the relaxation of a numeric planning domain and problem with
    all numeric components removed. Write the relaxed domain and problem to
    files and return the paths to the files.
    
    This will write to the same directory as the original PDDL files, but
    with the suffix '_relaxed' appended to the filename.

    Args:
        pddl_paths (Iterable[str]): Paths to PDDL files.

    Returns:
        Iterable[str]: Paths to the relaxed PDDL files.
    """
    relaxed_pddl_paths = []
    for path in pddl_paths:
        domains, problems = extract_all_domains_problems([path])
        
        domains = {domain_name: relax_numeric_domain(domain_hlist)
                for domain_name, domain_hlist in domains.items()}
        problems = {problem_name: relax_numeric_problem(problem_hlist)
                    for problem_name, problem_hlist in problems.items()}
    
        # there should only be one actual pddl file here, get its string
        assert len(domains) + len(problems) == 1

        pddl_str = None
        for domain_name in domains:
            pddl_str = hlist_to_sexprs(domains[domain_name], indent=4)
        for problem_name in problems:
            pddl_str = hlist_to_sexprs(problems[problem_name], indent=4)

        dir_name, file_name = os.path.split(path)
        file_name, file_ext = os.path.splitext(file_name)
        relaxed_path = os.path.join(dir_name, file_name + '_relaxed' + file_ext)

        with open(relaxed_path, 'w') as fp:
            fp.write(pddl_str)
        
        relaxed_pddl_paths.append(relaxed_path)
    
    return relaxed_pddl_paths


def replace_init_state(problem_hlist, tup_state):
    """Create modified hlist for problem that has old init state replaced with
    the given state, as generated by CanonicalState.to_tup_state()."""
    # check format for new atoms
    assert isinstance(tup_state, (tuple, list))
    for atom in tup_state:
        # make sure atoms have the right format (they should all be paren-free,
        # which is the same format used when interfacing with SSiPP or MDPSim)
        assert '(' not in atom and ')' not in atom, \
            "expecting atom format with no parens, but got '%s'" % (atom, )

    # build new problem hlist
    new_hlist = HList(parent=None)
    replaced_init = False
    for subsec in problem_hlist:
        if len(subsec) >= 1 and subsec[0] == ':init':
            init_hlist = HList(parent=new_hlist)
            init_hlist.append(":init")
            init_hlist.extend('(%s)' % atom for atom in tup_state[0])
            init_hlist.extend('(= (%s) %s)' % (flnt, val) 
                              for flnt, val in tup_state[1])
            new_hlist.append(init_hlist)
            replaced_init = True
        else:
            new_hlist.append(subsec)

    assert replaced_init, \
        "Could not find :init in hlist '%r'" % (problem_hlist, )

    return new_hlist
