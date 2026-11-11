import subprocess
import os
from ..util import check_path, get_directory_name, get_file_name
import sqlite3
class db:
    """
    This class handles specific tables and the corresponding queries
    """
    def __init__(self, nsys_report_path, output_path = None,):
        check_path(nsys_report_path)
        self.nsys = nsys_report_path
        self.output_path = output_path
        self.db = self.export_to_sqlite()
        assert self.db is not None
        self.cursor = self.db_connect()
    def run_command(self, nsys_path="nsys"):
        # handle the run command
        dir_name = get_directory_name(self.nsys)
        file_name = get_file_name(self.nsys)
        command = [nsys_path,"export","-t", "sqlite","--force-overwrite=true",self.nsys]
        if self.output_path is not None and self.output_path.lower() == "inherit":
            command = [nsys_path, "export", "-t","sqlite",
                        "--force-overwrite=true","-o",os.path.join(dir_name,file_name+".sqlite"),
                        self.nsys]
        result = subprocess.run(' '.join(command),
                                        shell=True,
                                        check=True,
                                        capture_output=True,
                                        text=True)
        return os.path.join(dir_name, file_name+".sqlite") if self.output_path.lower() == "inherit" else file_name+".sqlite"
    
    def export_to_json(self,):
        """
        Handles exporting to json file from nsys-rep format
        """
        pass
    
    def export_to_sqlite(self):
        """
        Handles exporting to sqlite format from nsys-rep format
        """
        try:
            return self.run_command(nsys_path="nsys")
            
        except subprocess.CalledProcessError:
            print("Initial attempt to run 'nsys' failed. Trying the environment variable path.")
            try:
                # get the nsys from the environment variable
                nsys_path = os.environ.get('nsys', None)
                if nsys_path is None:
                    raise RuntimeError("Couldn't find the nsys environment variable")
                return self.run_command(nsys_path=nsys_path)
            
            # except subprocess.CalledProcessError as err:
            #     # Handle errors in both attempts, providing detailed feedback.
            #     print(f"Command failed with return code {err.returncode}")
            #     print(f"Command output: {err.output}")
            #     print(f"Command stderr: {err.stderr}")
            #     raise RuntimeError(f"An error occurred while executing the command: {err}")
    
                
            except Exception as err:
                print(f"An error occured: {err}")
                # raise RuntimeError(f"Error occured: {err}")
                # try to find a sqlite file in the directory
                dir_name = get_directory_name(self.nsys)
                file_name = get_file_name(self.nsys)
                sqlite_file_name = os.path.join(dir_name,file_name+".sqlite") if self.output_path.lower() == "inherit" else file_name+".sqlite"

                if os.path.exists(sqlite_file_name):
                    print(f"{sqlite_file_name} exists")
                    return sqlite_file_name
                else:
                    raise RuntimeError(f"No sqlite file cannot be found: {sqlite_file_name}")
                    
    def db_connect(self,):
        connection = sqlite3.connect(self.db)
        return connection.cursor()
    
    def query(self, table_name, column_names=None):
        # by default it fetches all the columns
        try:
            if column_names in [[], None, "*"]:
                # fetch all info
                self.cursor.execute(f"SELECT * from {table_name}")
            else:
                columns = ','.join(column_names)
                print(columns)
                self.cursor.execute(f"SELECT {columns} from {table_name}")
        except Exception as err:
            raise RuntimeError(f"Error occured: {err}")
    @property
    def fetch_query(self):
        # return a table
        return self.cursor.fetchall()
    
    def get_stringIds_value(self,_id):
        """
        This method returns all the value with respect to the ids
        """
        self.cursor.execute(f"SELECT DISTINCT value FROM StringIds where id={_id}")
        return self.fetch_query # return an array
    # @property
    # def get_cursor_object(self,):
    #     return self.cursor
    
    # @property
    # def get_db_path(self,):
    #     return self.db