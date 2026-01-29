dtmc

module user_engagement
    // State encoding:
    // 0 = browsing (initial)
    // 1 = engaged
    // 2 = disengaged
    // 3 = converted (terminal - success)
    // 4 = abandoned (terminal - failure)
    
    state : [0..4] init 0;
    
    // From browsing: user sees initial prompt
    [] state=0 -> 0.70 : (state'=1)   // answers first question -> engaged
                + 0.20 : (state'=2)   // finds it irrelevant -> disengaged
                + 0.10 : (state'=4);  // closes immediately -> abandoned
    
    // From engaged: user is actively participating
    [] state=1 -> 0.50 : (state'=1)   // continues engaging
                + 0.30 : (state'=3)   // accepts recommendation -> converted
                + 0.15 : (state'=2)   // loses interest -> disengaged
                + 0.05 : (state'=4);  // suddenly leaves -> abandoned
    
    // From disengaged: user has lost interest
    [] state=2 -> 0.25 : (state'=1)   // re-engages
                + 0.30 : (state'=2)   // stays disengaged
                + 0.45 : (state'=4);  // gives up -> abandoned
    
    // Terminal states: self-loops
    [] state=3 -> 1.0 : (state'=3);   // converted
    [] state=4 -> 1.0 : (state'=4);   // abandoned

endmodule

// Labels for property specification
label "browsing" = (state=0);
label "engaged" = (state=1);
label "disengaged" = (state=2);
label "converted" = (state=3);
label "abandoned" = (state=4);
label "success" = (state=3);
label "failure" = (state=4);