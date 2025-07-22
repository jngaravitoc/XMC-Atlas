import sys
import yaml
import numpy as np

def create_example_paramfile(filename="param_example.yaml"):
    data = {
        'simulation': {
            'suite': "GC21",
            'sims_path': "/mnt/home/nico/ceph/gadget_runs/MWLMC/MWLMC5_b0/out/",
            'snapname': "MWLMC5_100M_b0_vir_OM3_G4",
            'initial_snapshot': 0,
            'final_snapshot': 10
        },
        'coefficients': {
            'coefspath': "/mnt/home/nico/ceph/projects/time-dependent-BFE/EXP/",
            'coefsname': "MWLMC5_b0_exp_np_10_8_nmax20_lmax_10"
        },
        'basis': {
            'basis_type': "Spherical",
            'nmax': 20,
            'lmax': 10,
            'mmax': 10,
            'rs': 40.85,
            'rhaloCut': 600
        }
    }

    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)



class BFEParameters:
    def __init__(self, paramfile):
        self.paramfile = paramfile   

    def simulation_parameters(self, paramfile):
        print("-> Loading simulation parameters") 
        required_params = [
                "suite", "sims_path", "snapname", 
                "initial_snapshot", "final_snapshot"
                ]   

        check = np.isin(required_params, list(paramfile["simulation"].keys()))   

        if False in check:
            print('!Error: required parameters {} not found in parameter file'.format(required_params))
            sys.exit(1)
        
        self.suite = paramfile["simulation"]["suite"]
        self.sims_path = paramfile["simulation"]["sims_path"]
        self.snapname = paramfile["simulation"]["snapname"]
        self.intial_snapshot = paramfile["simulation"]["initial_snapshot"]
        self.final_snapshot = paramfile["simulation"]["final_snapshot"]

    def coefficients_parameters(self, paramfile):
        print("-> Loading coefficients parameters") 
        required_params = [
                "coefspath", "coefsname"
                ]   

        check = np.isin(required_params, list(paramfile["coefficients"].keys()))   

        if False in check:
            print('!Error: required parameters {} not found in parameter file'.format(required_params))
            sys.exit(1)
        
        self.coefspath = paramfile["coefficients"]["coefspath"]
        self.coefsname = paramfile["coefficients"]["coefsname"]
    
    def basis_parameters(self, paramfile):
        print("-> Loading basis coefficients parameters")
        basis_params = paramfile["basis"].keys()
        required_params = ["basis_type"]

        check = np.isin(required_params, list(paramfile["basis"].keys()))   

        if False in check:
            print('!Error: required parameters {} not found in parameter file'.format(required_params))
            sys.exit(1)
        
        if paramfile["basis"]["basis_type"] == "Spherical":
            self.nmax = paramfile["basis"]["nmax"]
            self.lmax = paramfile["basis"]["lmax"]
            self.mmax = paramfile["basis"]["mmax"]
            self.rs = paramfile["basis"]["rs"]
            self.rhalo_cut = paramfile["basis"]["rhaloCut"]

        else:
            print("Error: Basis type is not yet implemented")
            sys.exit(1)

    def check_categories(self, paramfile):
        required_categories = ["simulation", "coefficients", "basis"]
        
        check = np.isin(required_categories, list(paramfile.keys()))
        if False in check:
            print('!Error: required categories {} not found in parameter file'.format(required_categories))
            sys.exit(1)

    def load_parameters(self):
        with open(self.paramfile) as f:
            d = yaml.safe_load(f)
        
        # Check categories and load parameters
        self.check_categories(d)
        self.simulation_parameters(d)
        self.coefficients_parameters(d)
        self.basis_parameters(d)

if __name__ == '__main__':
    """
    description="Parameters file for bfe-py")
    parser.add_argument(
        "--param", dest="paramFile", 
        default="config.yaml",
        type=str, 
        help="provide parameter file")

    args = parser.parse_args()
    """
    print("Running test on allvars functions:")
    paramfile = "test_param.yaml"
    create_example_paramfile(paramfile)
    test_reader = BFEParameters(paramfile)
    test_reader.load_parameters()
    
    assert test_reader.nmax == 20
    assert test_reader.lmax == 10
    assert test_reader.mmax == 10
    assert test_reader.rs == 40.85
    assert test_reader.rhalo_cut == 600
    assert test_reader.coefspath == "/mnt/home/nico/ceph/projects/time-dependent-BFE/EXP/"
    print("-> All test passed")
