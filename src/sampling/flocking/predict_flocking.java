/* package required for calls from parent folder */
package sampling.flocking;

/* libraries included */
import org.nlogo.headless.HeadlessWorkspace;
import java.io.*;
import java.util.*;

/* main class */
public class predict_flocking {

    // constants
    private static int RUNS = 1;
    private static int SAMPLES = 10;
    private static int TIME_LAPSE = 200;

    // return time via static function
    private static long getCurrentTimeInMillis() {
	return System.currentTimeMillis();
    }

    // main function
    // expects values of independent variables as input
    public static void main(String[] argv) {
	String folderName = "images" + "_" + String.valueOf(getCurrentTimeInMillis());

	// create temporary workspace and image output folder
	HeadlessWorkspace workspace = HeadlessWorkspace.newInstance();	
	boolean folderCreated = (new File("sampling/flocking/" + folderName)).mkdirs();

	try {
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


			// independent variables as input
			String k1 = alpValues.get(0);
			String k2 = alpValues.get(1);
			String k3 = alpValues.get(2);	

			// open and initialize model
			workspace.open("sampling/flocking/Flocking.nlogo");
			workspace.command("random-seed 0");
			workspace.command("set vision 10.0");
			workspace.command("set minimum-separation 1.0");   
			workspace.command("set population 300");

			// average over runs
			for(int run = 0; run < RUNS; run++) {            
				// set independent parameters
				workspace.command("set max-align-turn " + k1);
				workspace.command("set max-cohere-turn " + k2);
				workspace.command("set max-separate-turn " + k3);
				workspace.command("setup");
				workspace.command("repeat " + Integer.toString(TIME_LAPSE) + " [ go ]");

				workspace.command("save-image " + Integer.toString(run) + " \"" + folderName + "\"");

				// sample data
				for(int sample = 0; sample < SAMPLES; sample++) {
					// run one epoch
					workspace.command("go");
					workspace.command("save-image " + Integer.toString(run) + " \"" + folderName + "\"");
				}

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
