/* package required for calls from parent folder */
package sampling.wolf_sheep;

/* libraries included */
import org.nlogo.headless.HeadlessWorkspace;
import java.io.*;
import java.util.*;

public class predict_wolf_sheep {
    public static void main(String[] argv) {
        HeadlessWorkspace workspace =
            HeadlessWorkspace.newInstance() ;
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

            workspace.open("sampling/wolf_sheep/Wolf Sheep Predation.nlogo");

        	workspace.command("set grass? true");
            workspace.command("set initial-number-sheep 100");
            workspace.command("set initial-number-wolves 50");   
            
            	// wolf-reproduce
            	String k1 = alpValues.get(0);
            	
            	// sheep-reproduce
				String k2 = alpValues.get(1);
				
				//wolf-gain-from-food
				String k3 = alpValues.get(2);
				
				//sheep-gain-from-food
				String k4 = alpValues.get(3);
				
				//grass-regrowth-time
				String k5 = alpValues.get(4);


				workspace.command("set wolf-reproduce " + k1);
				workspace.command("set sheep-reproduce " + k2);
				workspace.command("set wolf-gain-from-food " + k3);
				workspace.command("set sheep-gain-from-food " + k4);
				workspace.command("set grass-regrowth-time " + k5);

            	workspace.command("setup");
            	workspace.command("repeat 500 [ go ]");
            	
            	String wolves = "" + workspace.report("count wolves");
            	String sheep = "" + workspace.report("count sheep");
            	
            	for(int j = 0; j < 750; j++) {
            		workspace.command("go");
            		wolves += "," + workspace.report("count wolves");
            		sheep += "," + workspace.report("count sheep");
            	}
            	
		String[] wolvesValues = wolves.split(",");
		float sum = 0;
		float count = 0;
		for(String value: wolvesValues){
			sum += Float.parseFloat(value);
			count += 1;
		}
		float wolvesAverage = sum/count;

		String[] sheepValues = sheep.split(",");
		sum = 0;
		count = 0;
		for(String value: sheepValues){
			sum += Float.parseFloat(value);
			count += 1;
		}
		float sheepAverage = sum/count;

            	System.out.print("[(" + String.valueOf(wolvesAverage) + ", " + String.valueOf(sheepAverage) + ")]\n");

            }
            
        	workspace.dispose();

        }
        catch(Exception ex) {
            ex.printStackTrace();
        }
    }
    
    
}
