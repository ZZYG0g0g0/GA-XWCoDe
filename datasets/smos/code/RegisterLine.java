package smos.bean;

import java.io.Serializable;

public class RegisterLine implements Serializable {

	/**
	 * Class modelling a log line
	 * @author Nicola Pisanti
	 * @version 1.0 
	 */
	private static final long serialVersionUID = -6010085925185873838L;
	
	private UserListItem student;
	private Absence absence;
	private Delay delay;
	
	public RegisterLine(){
		absence=null;
		delay=null;
		
	}
	
	/**
	 * Method that returns the student of this log line
	 * @return a User object representing the student
	 */
	public UserListItem getStudent() {
		return student;
	}
	/**
	 * Method that sets the student to this log line
	 * @param a User object representing the student to enter
	 */
	public void setStudent(UserListItem student) {
		this.student = student;
	}
	/**
	 * Method that returns the student's absence from this log line
* 	 @return an Absence object representing absence, or null if the student was present
	 */
	public Absence getAbsence() {
		return absence;
	}
	/**
	 * Method that sets the student's absence from this log line
	 * @param an object of type Absence to set
	 */
	public void setAbsence(Absence absence) {
		this.absence = absence;
	}
	/**
	 * Method that returns the student's delay in this log line
	 * @return a Delay object representing the delay, or null if the student arrived on time or was absent
	 */
	public Delay getDelay() {
		return delay;
	}
	/**
	 * Method setting the student's delay in this log line
	 * @param a Delay object to set
	 */
	public void setDelay(Delay delay) {
		this.delay = delay;
	}

}
