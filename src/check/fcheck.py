import os.path
import re
import subprocess


FCHECK4_ERROR_HEADER = \
    b'******************************************** ERRORS ***********************************************'
FCHECK4_WARNING_HEADER = \
    b'******************************************* WARNINGS **********************************************'
FCHECK4_SUMMARY = \
    (b'####################################################################################################\n'
     b'SCAN SUMMARY')
FCHECK4_SEP = \
    b'####################################################################################################'


def run_fcheck(filename, ini_file=None):
    """
    Check file format complies to SeaBASS with fcheck4 PERL script.

    """
    fcheck_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fcheck4', 'fcheck4.pl')
    out = subprocess.run([fcheck_script, filename], capture_output=True)
    sections = re.split(b'(' +  b'|'.join([re.escape(FCHECK4_ERROR_HEADER),
                                           re.escape(FCHECK4_WARNING_HEADER),
                                           re.escape(FCHECK4_SUMMARY),
                                           re.escape(FCHECK4_SEP)]) + b')', out.stdout)
    i, success, errors, warnings, error_types, warning_types = 0, None, [], [], [], []
    while i < len(sections):
        if sections[i] == FCHECK4_SUMMARY:
            s = sections[i + 1]
            i += 2
            lines = s.split(b'\n\n', 2)
            for l in lines[1:3]:
                if b'errors' in l:
                    error_types = re.split(b'\n', l)[1:]
                elif b'warnings' in l:
                    warning_types = re.split(b'\n', l)[1:]
        elif sections[i] == FCHECK4_SEP:
            s = sections[i + 1]
            i += 2
            l = s.split(b'\n', 2)
            # Check Filename
            if l[1].decode() != filename:
                raise ValueError(f'{os.path.basename(filename)}: Filename mismatch in fcheck4 output.')
            # Get Pass/Fail
            if b'This file passed the FCHECK.' in l[2] and success is None:
                success = True
            elif b'This file failed the FCHECK.' in l[2]:
                success = False
            else:
                raise ValueError(f'{os.path.basename(filename)}: Unable to parse fcheck4 output.')
        elif sections[i] == FCHECK4_ERROR_HEADER:
            s = sections[i + 1]
            i += 2
            errors = re.split(b'(\n[0-9]+\\))', s)[2::2]
        elif sections[i] == FCHECK4_WARNING_HEADER:
            s = sections[i + 1]
            i += 2
            warnings = re.split(b'(\n[0-9]+\\))', s)[2::2]
        else:
            i += 1
    if errors and success:
        raise ValueError(f'{os.path.basename(filename)}: Conflicting answer parsing fcheck4 output.')
    return success, errors, warnings, error_types, warning_types


if __name__ == "__main__":
    # Example usage
    result = run_fcheck('/Users/nils/Downloads/Seabass_Submission_20250321 3/PVST_VDIUP-Argo-Kd_1902601_202308_R1.sb')
    # result = run_fcheck('/Users/nils/Documents/Lab/Projects/VDIUP/_Floats/submitted/20250325/PVST_VDIUP-Argo-Kd_1902601_202304_R1.sb')
