/* package required for calls from parent folder */
package sampling.aids;

/* libraries included */
import org.nlogo.headless.HeadlessWorkspace;
import java.io.*;
import java.util.*;

public class predict_aids {

    public static void main(String[] argv) {
        HeadlessWorkspace workspace = HeadlessWorkspace.newInstance() ;

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

		workspace.open("sampling/aids/AIDS.nlogo");
		workspace.command("random-seed 0");
		workspace.command("set initial-people 300");
		workspace.command("set average-coupling-tendency 3");
		workspace.command("set average-commitment 2");   

		// average-condom-use
		String k1 = alpValues.get(0);

		// average-test-frequency
		String k2 = alpValues.get(1);

		workspace.command("set average-condom-use " + k1);
		workspace.command("set average-test-frequency " + k2);

		workspace.command("setup");

		String hivminus = "" + workspace.report("count turtles with [not infected?]");
		String hivplus = "" + workspace.report("count turtles with [known?]");

		while (Double.parseDouble("" + workspace.report("count turtles with [infected?] - count turtles with [known?]")) > 0.0) {
		workspace.command("go");
		hivminus += "," + workspace.report("count turtles with [not infected?]");
		hivplus += "," + workspace.report("count turtles with [known?]");
		}

		String[] hivminusValues = hivminus.split(",");
		float sum = 0;
		float count = 0;
		for(String value: hivminusValues){
			sum += Float.parseFloat(value);
			count += 1;
		}
		float hivminusAverage = sum/count;

		String[] hivplusValues = hivplus.split(",");
		sum = 0;
		count = 0;
		for(String value: hivplusValues){
			sum += Float.parseFloat(value);
			count += 1;
		}
		float hivplusAverage = sum/count;

		System.out.print("[(" + String.valueOf(hivminusAverage) + ", " + String.valueOf(hivplusAverage) + ")]\n");

		workspace.dispose();

	}
	}
	catch(Exception ex) {
	ex.printStackTrace();
	}
    }   
}
