/* package required for calls from parent folder */
package sampling.turbulence;

/* libraries included */
import org.nlogo.headless.HeadlessWorkspace;
import java.io.File;
import java.io.*;
import java.util.*;

/* main class */
public class predict_turbulence {

    // constants
    private static int INDEPENDENT_VARIABLES = 2;
    private static int RUNS = 1;
    private static int SAMPLES = 10;
    private static int TIME_LAPSE = 200;

	// return time via static function
    private static long getCurrentTimeInMillis() {
	return System.currentTimeMillis();
    }

    // main function
    // expects number of dependent variables as configuration parameter
    public static void main(String[] argv) {
	String folderName = "images" + "_" + String.valueOf(getCurrentTimeInMillis());

	// create temporary workspace and image output folder
	HeadlessWorkspace workspace = HeadlessWorkspace.newInstance();
	boolean folderCreated = (new File("sampling/turbulence/" + folderName)).mkdirs();
	
	// read environment variable
	int dependentVariables = Integer.parseInt(System.getenv("NUM_DEPENDENT"));

	try {
	    // open and initialize model
	    workspace.open("sampling/turbulence/Turbulence.nlogo");
	    
		// evaluate to list
		String alpConfigurationListString = argv[0];
		List<String> alpConfigurationList = new ArrayList<String>(Arrays.asList(alpConfigurationListString.split("\\(")));

		// skip first entry, which is empty
		for (int i = 1; i < alpConfigurationList.size(); i++) {
		String alpConfiguration = alpConfigurationList.get(i);

		// clean string
		alpConfiguration = alpConfiguration.replace(")", "").replace("]", "");

		// extract variables
		ArrayList<String> alpValues = new ArrayList<String>(Arrays.asList(alpConfiguration.split(",")));

	    // write header depending on number of independent variables
	    String headerString = "";
	    for(int counter = 0; counter < INDEPENDENT_VARIABLES; counter++) {
		headerString += "i";
	    }
	    for(int counter = 0; counter < dependentVariables; counter++) {
		headerString += "d";
	    }
	    System.out.println(headerString);
		
	    // collect data over random runs
	    for(int run = 0; run < RUNS; run++) {
            
		String k1 = alpValues.get(0);
		String k2 = alpValues.get(1);
		
		// set independent parameters
		workspace.command("setup-random");
		workspace.command("reset-ticks");
		workspace.command("set coupling-strength " + k1);
		workspace.command("set roughness " + k2);
		//workspace.command("random-seed 0");
		workspace.command("set initial-turbulence 50");   
		workspace.command("set auto-continue? false");

		workspace.command("repeat " + Integer.toString(TIME_LAPSE) + " [ go ]");
            	
		System.out.print(k1 + " " + k2 + " ");
		//workspace.command("save-image " + Integer.toString(run) + " \"" + "images" + "\"");

		// sample data
		for(int sample = 0; sample < SAMPLES; sample++) {
		    // run one epoch
		    //workspace.command("go");
		    //workspace.command("save-image " + Integer.toString(run) + " \"" + "images" + "\"");
		}
            	
		//workspace.command("go");
		workspace.command("save-image " + Integer.toString(run) + " \"" + folderName + "\"");
		System.out.print("\n");
		System.err.println(Integer.toString(run+1) + "/" + Integer.toString(RUNS));
	    }
		}
	    // close temporary workspace
	    workspace.dispose();
	}
        catch(Exception ex) {
	    // default
            ex.printStackTrace();
        }
    }
}
