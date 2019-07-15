/* package required for calls from parent folder */
package flocking;

/* libraries included */
import org.nlogo.headless.HeadlessWorkspace;
import java.io.File;

/* main class */
public class sample_flocking {

    // constants
    private static int INDEPENDENT_VARIABLES = 3;
    private static int RUNS = 10000;
    private static int SAMPLES = 10;
    private static int TIME_LAPSE = 200;

    // main function
    // expects number of dependent variables as configuration parameter
    public static void main(String[] argv) {
	// create temporary workspace and image output folder
	HeadlessWorkspace workspace = HeadlessWorkspace.newInstance();
	boolean folderCreated = (new File("flocking/images")).mkdirs();
	
	// read environment variable
	int dependentVariables = Integer.parseInt(System.getenv("NUM_DEPENDENT"));

	try {
	    // open and initialize model
	    workspace.open("flocking/Flocking.nlogo");
	    workspace.command("random-seed 0");
	    workspace.command("set vision 10.0");
	    workspace.command("set minimum-separation 1.0");   
	    workspace.command("set population 300");
	    
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
            
		String k1 = Double.toString(Math.random() * (20.0 - 1.0) + 1.0);
		String k2 = Double.toString(Math.random() * (20.0 - 1.0) + 1.0);
		String k3 = Double.toString(Math.random() * (20.0 - 1.0) + 1.0);
		
		// set independent parameters
		workspace.command("set max-align-turn " + k1);
		workspace.command("set max-cohere-turn " + k2);
		workspace.command("set max-separate-turn " + k3);
		workspace.command("setup");
		workspace.command("repeat " + Integer.toString(TIME_LAPSE) + " [ go ]");
            	
		System.out.print(k1 + " " + k2 + " " + k3 + " ");
		workspace.command("save-image " + Integer.toString(run) + " \"" + "images" + "\"");

		// sample data
		for(int sample = 0; sample < SAMPLES; sample++) {
		    // run one epoch
		    workspace.command("go");
		    workspace.command("save-image " + Integer.toString(run) + " \"" + "images" + "\"");
		}
            	
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
