/* package required for calls from parent folder */
package turbulence;

/* libraries included */
import org.nlogo.headless.HeadlessWorkspace;
import java.io.File;

/* main class */
public class sample_turbulence {

    // constants
    private static int INDEPENDENT_VARIABLES = 2;
    private static int RUNS = 10000;
    private static int SAMPLES = 10;
    private static int TIME_LAPSE = 200;

    // main function
    // expects number of dependent variables as configuration parameter
    public static void main(String[] argv) {
	// create temporary workspace and image output folder
	HeadlessWorkspace workspace = HeadlessWorkspace.newInstance();
	boolean folderCreated = (new File("turbulence/images")).mkdirs();
	
	// read environment variable
	int dependentVariables = Integer.parseInt(System.getenv("NUM_DEPENDENT"));

	try {
	    // open and initialize model
	    workspace.open("turbulence/Turbulence.nlogo");
	    
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
            
		String k1 = Double.toString(Math.random() * (1.0) + 0.0);
		String k2 = Double.toString(Math.random() * (0.0250) + 0.0);
		
		// set independent parameters
		workspace.command("setup-random");
		workspace.command("reset-ticks");
		workspace.command("set coupling-strength " + k1);
		workspace.command("set roughness " + k2);
		workspace.command("random-seed 0");
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
		workspace.command("save-image " + Integer.toString(run) + " \"" + "images" + "\"");
		System.out.print("\n");
		System.err.println(Integer.toString(run+1) + "/" + Integer.toString(RUNS));
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
