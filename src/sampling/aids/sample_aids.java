/* package required for calls from parent folder */
package aids;

/* libraries included */
import org.nlogo.headless.HeadlessWorkspace;

public class sample_aids {
	// constants
	private static int RUNS = 10000;

    public static void main(String[] argv) {
        HeadlessWorkspace workspace = HeadlessWorkspace.newInstance() ;
        try {
            workspace.open("aids/AIDS.nlogo");

       	    workspace.command("set initial-people 300");
            workspace.command("set average-coupling-tendency 3");
            workspace.command("set average-commitment 2");   
            
            System.out.println("iidd");
            
            for(int i = 0; i < RUNS; i++) {
            
            	// average-condom-use
            	String k1 = Double.toString(Math.random() * (10.0 - 0.0) + 0.0);
            	
            	// average-test-frequency
		String k2 = Double.toString(Math.random() * (1.0 - 0.0) + 0.0);

		workspace.command("set average-condom-use " + k1);
		workspace.command("set average-test-frequency " + k2);

            	workspace.command("setup");
            	
            	System.out.print(k1 + " " + k2 + " ");
            	String hivminus = "" + workspace.report("count turtles with [not infected?]");
            	String hivplus = "" + workspace.report("count turtles with [known?]");
            	
            	while (Double.parseDouble("" + workspace.report("count turtles with [infected?] - count turtles with [known?]")) > 0.0) {
            		workspace.command("go");
            		hivminus += "," + workspace.report("count turtles with [not infected?]");
            		hivplus += "," + workspace.report("count turtles with [known?]");
            	}
            	
            	System.out.print(hivminus + " " + hivplus + "\n");
            	System.err.println(Integer.toString(i) + "/10000");
            }
            
        	workspace.dispose();

        }
        catch(Exception ex) {
            ex.printStackTrace();
        }
    }
    
    
}
