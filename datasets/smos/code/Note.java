package smos.bean;

import java.io.Serializable;
import java.util.Date;

/**
 * Class modelling a note on the log
 * @author Nicola Pisanti
 * @version 1.0
 */


public class Note implements Serializable{

	private static final long serialVersionUID = 5953926210895315436L;
	
	private int idNote;
	private int idUser;
	private Date dateNote;
	private String description;
	private String teacher;
	private int academicYear;
	
	
	public Note(){
		
	}
	
	
	
	/**
	 * Method that returns the note id
     * @return an integer representing the note id
	 */
	public int getIdNote() {
		return idNote;
	}
	/**
	 * Method to set the note id
	 * @param an integer representing the new id value
	 */
	public void setIdNote(int pIdNote) {
		this.idNote = pIdNote;
	}
	/**
	 * Method that returns the id of the student who received the note
	 * @return the id of the user who received the note
	 */
	public int getIdUser() {
		return idUser;
	}
	/**
	 * Method to set the id of the student who received the note
	 * @param an integer representing the new id value
	 */
	public void setIdUser(int pIdUser) {
		this.idUser = pIdUser;
	}
	/**
	 * Method that returns a string representing the date the note was given
	 * @return a string representing the date of the note
	 */
	public Date getDateNote() {
		return dateNote;
	}
	/**
	 * Method that sets a string representing the date the note was given
  	 * @param the string representing the new date
	 */
	public void setDateNote(Date pDateNote) {
		this.dateNote = pDateNote;
	}
	/**
	 * Method that returns note text
	 * @return a string representing the note text
	 */
	public String getDescription() {
		return description;
	}
	/**
	 * Method setting the description of the note
	 * @param a string containing the description of the note
	 */
	public void setDescription(String pDescription) {
		this.description = pDescription;
	}
	/**
	 * Method that returns the id of the teacher who gave the note
	 * @return an integer representing the teacher id
	 */
	public String getTeacher() {
		return teacher;
	}
	/**
	 * Method setting the id of the teacher who gave the note
	 * @param teacher the teacher to set
	 */
	public void setTeacher(String pTeacher) {
		this.teacher = pTeacher;
	}
	/**
	 * Method that returns the current academic year
	 * @return an integer indicating the year the lessons start
	 */
	public int getAcademicYear() {
		return academicYear;
	}
	/**
	 * Medoto setting the current academic year during the assignment of the note
	 * @param an integer indicating the start year of the lessons to be entered
	 */
	public void setAcademicYear(int academicYear) {
		this.academicYear = academicYear;
	}

	
	
	
	
	
}
