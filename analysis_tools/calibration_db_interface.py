import json
import os 
import requests

class CalibrationDBInterface:
    
    def __init__(self, credential_path="./.wctecaldb.credential", calibration_db_url = "https://wcte.caldb.triumf.ca/api/v1/"):
        self.credential_path = credential_path
        self.calibration_db_url = calibration_db_url
        self.get_jwt_token()
        
    def get_jwt_token(self):
        print("Initialise Calibration Database Authentication")
        token_url = self.calibration_db_url+"login"
        # Check if the credential file exists and is readable
        if not os.path.isfile(self.credential_path) or not os.access(self.credential_path, os.R_OK):
            print("Can't find credential path at ",self.credential_path)
            print("See instructions https://wcte.hyperk.ca/documents/calibration-db-apis/v1-api-endpoints-documentation")
            print("Or copy credential file from EOS 'cp /eos/experiment/wcte/calibration_db_credentials/.wctecaldb.credential .' ")
            
            raise FileNotFoundError(f"Credential file not found or not readable: {self.credential_path}")

        # Read credentials from the file (expects shell-style exports or var=val lines)
        credentials = {}
        with open(self.credential_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                credentials[key.strip()] = value.strip().strip('"').strip("'")

        username = credentials.get('WCTECALDB_USERNAME')
        password = credentials.get('WCTECALDB_PASSWORD')

        if not username or not password:
            print("See instructions https://wcte.hyperk.ca/documents/calibration-db-apis/v1-api-endpoints-documentation")
            raise ValueError("WCTECALDB_USERNAME or WCTECALDB_PASSWORD not found in the credentials file.")

        # Make the POST request to get the token
        response = requests.post(
            token_url,
            headers={'Content-Type': 'application/json'},
            data=json.dumps({
                'username': username,
                'password': password
            })
        )
        if response.status_code != 201:
            print(response)
            raise ValueError(f"Unexpected status code {response.status_code}, expected 201.")
        
        # Parse the token from the response
        self.jwt_token = response.json().get('access_token')
        print(self.jwt_token)
        if not self.jwt_token:
            raise ValueError("Failed to retrieve access_token from the response.")

        print("Token successfully retrieved.")
        return self.jwt_token
    
    def print_jwt_token(self):
        print(self.jwt_token)
        
    
    def get_calibration_constants(self, run_number, time, calibration_name, official):
        url = self.calibration_db_url+"calibration_constants/by_validity_period"
        # params = {
        #     "run_number": run_number,
        #     "time": time,
        #     "calibration_name": calibration_name,
        #     "official":official
        # }
        params = {
            "run_number": run_number,
            "time": time,
            "calibration_name": calibration_name,
            "official": official
        }
        headers = {
            "Authorization": f"Bearer {self.jwt_token}"
        }

        response = requests.get(url, params=params, headers=headers)
                
        if response.status_code != 200:
            print(response, response.json())
            raise ValueError(f"Unexpected status code in constant request response {response.status_code}, expected 201. \n"+
                            str(response.json())
                            )
        
        calibration_data = response.json()

        timing_offsets_list = calibration_data[0]['data']
        revision_id = calibration_data[0]['revision_id']
        insert_time = calibration_data[0]['insert_time']
        return timing_offsets_list, revision_id, insert_time

    def post_calibration_constants(self, calibration_name, calibration_method, data,
                                   start_run_number, end_run_number,
                                   start_time=0, end_time=1, official=False):
        """POST a list of calibration constants to the database.

        Parameters
        ----------
        calibration_name : str
            Name of the calibration (e.g. 'pmt_state').
        calibration_method : str
            Method label (e.g. 'pmt_state').
        data : list of dict
            Per-channel entries, e.g. [{"glb_pmt_id": 203, "pmt_status": "OFFLINE"}].
        start_run_number : int
            First run number for which these constants are valid.
        end_run_number : int
            Last run number for which these constants are valid.
        start_time : int, optional
            Unix timestamp validity start (default 0).
        end_time : int, optional
            Unix timestamp validity end (default 0).
        official : bool, optional
            Whether this is an official calibration entry (default False).

        Returns
        -------
        dict
            Parsed JSON response from the server.
        """
        url = self.calibration_db_url + "calibration_constants"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jwt_token}"
        }
        payload = {
            "start_run_number": start_run_number,
            "end_run_number": end_run_number,
            "start_time": start_time,
            "end_time": end_time,
            "calibration_name": calibration_name,
            "calibration_method": calibration_method,
            "official": official,
            "data": data
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code not in (200, 201):
            print(response, response.text)
            raise ValueError(
                f"Unexpected status code {response.status_code} when posting calibration constants.\n"
                + response.text
            )
        print(f"Successfully posted {len(data)} entries for '{calibration_name}' "
              f"(runs {start_run_number}–{end_run_number}).")
        return response.json()

    def post_bad_pmts(self, bad_pmts, run_number, pmt_status="OFFLINE", official=True):
        """Upload a list of problematic PMTs to the DB as pmt_state entries for a single run.

        Parameters
        ----------
        bad_pmts : list of int
            Global PMT IDs (slot*100 + position) to mark, e.g. slot 2 pos 3 -> 203.
        run_number : int
            Run for which these channels are bad (used as both start and end run).
        pmt_status : str, optional
            Status string to assign (default 'OFFLINE').
        official : bool, optional
            Whether this is an official calibration entry (default True).

        Returns
        -------
        dict
            Response from post_calibration_constants.
        """
        data = [{"glb_pmt_id": int(ch), "pmt_status": pmt_status}
                for ch in bad_pmts]
        return self.post_calibration_constants(
            calibration_name="pmt_state",
            calibration_method="pmt_state",
            data=data,
            start_run_number=int(run_number),
            end_run_number=int(run_number),
            official=official,
        )

    def get_bad_pmts(self, run_number):
        """Query the database for manually-identified bad PMTs for a given run.

        Returns the list of global PMT IDs (slot*100 + position) that have been
        stored in the database as 'pmt_state' = OFFLINE for this run.

        Parameters
        ----------
        run_number : int or str

        Returns
        -------
        np.ndarray of int
            Global PMT IDs of bad PMTs. Empty array if none are stored.
        """
        import numpy as np
        try:
            pmt_state_data, _, _ = self.get_calibration_constants(
                run_number=int(run_number), time=0,
                calibration_name="pmt_state", official=1
            )
            return np.array([entry["glb_pmt_id"] for entry in pmt_state_data
                             if entry.get("pmt_status_id") == 2], dtype=int)  # 2 = OFFLINE
        except Exception:
            return np.array([], dtype=int)